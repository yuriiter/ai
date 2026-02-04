package rag

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	openai "github.com/sashabaranov/go-openai"
	"github.com/taylorskalyo/goreader/epub"
	"github.com/yuriiter/ai/pkg/ui"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type OpenAIEmbedder struct {
	client *openai.Client
	model  string
}

func (o *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := o.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: openai.EmbeddingModel(o.model),
	})
	if err != nil {
		return nil, err
	}
	var res [][]float32
	for _, d := range resp.Data {
		res = append(res, d.Embedding)
	}
	return res, nil
}

type LocalEmbedder struct {
	interfaceModel textencoding.Interface
}

func NewLocalEmbedder() (*LocalEmbedder, error) {
	fmt.Printf("%sInitializing local embedding model (downloading if needed)...%s\n", ui.ColorBlue, ui.ColorReset)

	model, err := tasks.Load[textencoding.Interface](&tasks.Config{
		ModelsDir: filepath.Join(os.Getenv("HOME"), ".cybertron"),
		ModelName: "sentence-transformers/all-MiniLM-L6-v2",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load local model: %w", err)
	}
	return &LocalEmbedder{interfaceModel: model}, nil
}

func (l *LocalEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	var results [][]float32
	for _, text := range texts {
		output, err := l.interfaceModel.Encode(ctx, text, int(bert.MeanPooling))
		if err != nil {
			return nil, err
		}

		vec := output.Vector.Data().F32()
		results = append(results, vec)
	}
	return results, nil
}

type Chunk struct {
	Text     string
	Filename string
	Vector   []float32
}

type Engine struct {
	embedder Embedder
	Chunks   []Chunk
}

func New(client *openai.Client, configProvider string, configModel string) (*Engine, error) {
	var emb Embedder
	var err error

	if configProvider == "local" {
		emb, err = NewLocalEmbedder()
	} else {
		emb = &OpenAIEmbedder{client: client, model: configModel}
	}

	if err != nil {
		return nil, err
	}

	return &Engine{
		embedder: emb,
		Chunks:   make([]Chunk, 0),
	}, nil
}

func (e *Engine) IngestGlob(ctx context.Context, globPattern string) error {
	files := findFiles(globPattern)
	if len(files) == 0 {
		return fmt.Errorf("no files found matching %s", globPattern)
	}

	fmt.Printf("%sRAG: Found %d files. Processing...%s\n", ui.ColorBlue, len(files), ui.ColorReset)

	var textsToEmbed []string
	var mapIndexToMeta []struct {
		Text     string
		Filename string
	}

	for _, file := range files {
		content, err := extractText(file)
		if err != nil {
			continue
		}
		if strings.TrimSpace(content) == "" {
			continue
		}

		chunks := chunkText(content, 1000, 200)
		for _, c := range chunks {
			textsToEmbed = append(textsToEmbed, c)
			mapIndexToMeta = append(mapIndexToMeta, struct {
				Text     string
				Filename string
			}{Text: c, Filename: file})
		}
	}

	if len(textsToEmbed) == 0 {
		return fmt.Errorf("no text content extracted")
	}

	fmt.Printf("Generating embeddings for %d chunks...\n", len(textsToEmbed))

	batchSize := 20
	for i := 0; i < len(textsToEmbed); i += batchSize {
		end := i + batchSize
		if end > len(textsToEmbed) {
			end = len(textsToEmbed)
		}

		batch := textsToEmbed[i:end]
		vectors, err := e.embedder.Embed(ctx, batch)
		if err != nil {
			return fmt.Errorf("embedding error: %w", err)
		}

		for j, vec := range vectors {
			meta := mapIndexToMeta[i+j]
			e.Chunks = append(e.Chunks, Chunk{
				Text:     meta.Text,
				Filename: meta.Filename,
				Vector:   vec,
			})
		}
		fmt.Printf(".")
	}
	fmt.Println("\nDone.")

	return nil
}

func (e *Engine) Search(ctx context.Context, query string, topK int) ([]Chunk, error) {
	vectors, err := e.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("failed to embed query")
	}

	queryVector := vectors[0]

	type scoredChunk struct {
		Chunk Chunk
		Score float64
	}

	var scores []scoredChunk
	for _, chunk := range e.Chunks {
		score := cosineSimilarity(queryVector, chunk.Vector)
		scores = append(scores, scoredChunk{Chunk: chunk, Score: score})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	if len(scores) < topK {
		topK = len(scores)
	}

	var results []Chunk
	for i := 0; i < topK; i++ {
		results = append(results, scores[i].Chunk)
	}

	return results, nil
}

func findFiles(pattern string) []string {
	var files []string
	if strings.Contains(pattern, "**") {
		parts := strings.Split(pattern, "**")
		rootDir := "."
		if parts[0] != "" {
			rootDir = parts[0]
		}
		suffix := strings.TrimPrefix(pattern, rootDir)
		suffix = strings.TrimPrefix(suffix, "**")
		suffix = strings.TrimPrefix(suffix, string(filepath.Separator))

		filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
			if err == nil && !d.IsDir() {
				match, _ := filepath.Match(suffix, filepath.Base(path))
				if suffix == "" || match || strings.HasSuffix(filepath.Base(path), strings.TrimPrefix(suffix, "*")) {
					files = append(files, path)
				}
			}
			return nil
		})
	} else {
		f, _ := filepath.Glob(pattern)
		files = f
	}
	return files
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func chunkText(text string, chunkSize, overlap int) []string {
	var chunks []string
	runes := []rune(text)
	if len(runes) == 0 {
		return chunks
	}
	for i := 0; i < len(runes); i += (chunkSize - overlap) {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}

func extractText(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".txt", ".md", ".go", ".js", ".json", ".py", ".html", ".css", ".java", ".c", ".h", ".cpp":
		b, err := os.ReadFile(path)
		return string(b), err
	case ".pdf":
		f, r, err := pdf.Open(path)
		if err != nil {
			return "", err
		}
		defer f.Close()
		var sb strings.Builder
		for i := 1; i <= r.NumPage(); i++ {
			p := r.Page(i)
			if !p.V.IsNull() {
				t, _ := p.GetPlainText(nil)
				sb.WriteString(t + "\n")
			}
		}
		return sb.String(), nil
	case ".docx":
		return parseDocx(path)
	case ".epub":
		rc, err := epub.OpenReader(path)
		if err != nil {
			return "", err
		}
		defer rc.Close()
		var sb strings.Builder
		for _, item := range rc.Rootfiles[0].Manifest.Items {
			if strings.Contains(item.MediaType, "html") {
				f, _ := item.Open()
				b, _ := io.ReadAll(f)
				f.Close()
				sb.WriteString(stripTags(string(b)) + "\n")
			}
		}
		return sb.String(), nil
	}
	return "", fmt.Errorf("unsupported type")
}

func parseDocx(path string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	var sb strings.Builder
	for _, f := range r.File {
		if f.Name == "word/document.xml" {
			rc, _ := f.Open()
			defer rc.Close()
			dec := xml.NewDecoder(rc)
			for {
				t, _ := dec.Token()
				if t == nil {
					break
				}
				if se, ok := t.(xml.StartElement); ok && se.Name.Local == "t" {
					var s string
					dec.DecodeElement(&s, &se)
					sb.WriteString(s)
				}
				if se, ok := t.(xml.StartElement); ok && se.Name.Local == "p" {
					sb.WriteString("\n")
				}
			}
		}
	}
	return sb.String(), nil
}

func stripTags(c string) string {
	var sb strings.Builder
	in := false
	for _, r := range c {
		if r == '<' {
			in = true
			continue
		}
		if r == '>' {
			in = false
			continue
		}
		if !in {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}
