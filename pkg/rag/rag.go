package rag

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

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
	const maxBatchSize = 2048
	var allResults [][]float32

	for i := 0; i < len(texts); i += maxBatchSize {
		end := i + maxBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		resp, err := o.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
			Input: batch,
			Model: openai.EmbeddingModel(o.model),
		})
		if err != nil {
			return nil, err
		}

		for _, d := range resp.Data {
			allResults = append(allResults, d.Embedding)
		}
	}
	return allResults, nil
}

type LocalEmbedder struct {
	interfaceModel textencoding.Interface
	mu             sync.Mutex
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
	results := make([][]float32, len(texts))

	numWorkers := 4
	if len(texts) < numWorkers {
		numWorkers = len(texts)
	}

	type job struct {
		index int
		text  string
	}

	jobs := make(chan job, len(texts))
	errors := make(chan error, numWorkers)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				output, err := l.interfaceModel.Encode(ctx, j.text, int(bert.MeanPooling))
				if err != nil {
					select {
					case errors <- err:
					default:
					}
					return
				}

				vec := output.Vector.Data().F32()

				l.mu.Lock()
				results[j.index] = vec
				l.mu.Unlock()
			}
		}()
	}

	for i, text := range texts {
		jobs <- job{index: i, text: text}
	}
	close(jobs)

	wg.Wait()
	close(errors)

	if err := <-errors; err != nil {
		return nil, err
	}

	return results, nil
}

type Chunk struct {
	Text     string
	Filename string
	Vector   []float32
}

type FileMetadata struct {
	Path    string
	ModTime time.Time
	Size    int64
}

type EmbeddingCache struct {
	Chunks       []Chunk
	GlobPattern  string
	Provider     string
	Model        string
	Version      int
	CreatedAt    time.Time
	FileMetadata []FileMetadata
	ContentHash  string
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

func calculateContentHash(files []string) (string, error) {
	hasher := sha256.New()

	sortedFiles := make([]string, len(files))
	copy(sortedFiles, files)
	sort.Strings(sortedFiles)

	for _, file := range sortedFiles {
		data, err := os.ReadFile(file)
		if err != nil {
			return "", err
		}
		hasher.Write([]byte(file))
		hasher.Write(data)
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func getFileMetadata(files []string) ([]FileMetadata, error) {
	var metadata []FileMetadata

	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			return nil, err
		}

		metadata = append(metadata, FileMetadata{
			Path:    file,
			ModTime: info.ModTime(),
			Size:    info.Size(),
		})
	}

	return metadata, nil
}

func (e *Engine) ValidateCache(cachePath string, globPattern string) (bool, string) {
	file, err := os.Open(cachePath)
	if err != nil {
		return false, "cache file not found"
	}
	defer file.Close()

	var cache EmbeddingCache
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&cache); err != nil {
		return false, "failed to decode cache"
	}

	if cache.GlobPattern != globPattern {
		return false, fmt.Sprintf("pattern mismatch: cached='%s' vs current='%s'", cache.GlobPattern, globPattern)
	}

	currentFiles := findFiles(globPattern)
	if len(currentFiles) == 0 {
		return false, "no files found matching pattern"
	}

	if len(currentFiles) != len(cache.FileMetadata) {
		return false, fmt.Sprintf("file count changed: cached=%d vs current=%d", len(cache.FileMetadata), len(currentFiles))
	}

	currentMetadata, err := getFileMetadata(currentFiles)
	if err != nil {
		return false, "failed to read current file metadata"
	}

	cachedMap := make(map[string]FileMetadata)
	for _, m := range cache.FileMetadata {
		cachedMap[m.Path] = m
	}

	for _, current := range currentMetadata {
		cached, exists := cachedMap[current.Path]
		if !exists {
			return false, fmt.Sprintf("new file detected: %s", current.Path)
		}

		if !current.ModTime.Equal(cached.ModTime) || current.Size != cached.Size {
			return false, fmt.Sprintf("file changed: %s (time or size mismatch)", current.Path)
		}
	}

	return true, ""
}

func (e *Engine) SaveEmbeddings(filepath string, globPattern string, provider string, model string) error {
	files := findFiles(globPattern)
	metadata, err := getFileMetadata(files)
	if err != nil {
		return fmt.Errorf("failed to get file metadata: %w", err)
	}

	contentHash, err := calculateContentHash(files)
	if err != nil {
		return fmt.Errorf("failed to calculate content hash: %w", err)
	}

	cache := EmbeddingCache{
		Chunks:       e.Chunks,
		GlobPattern:  globPattern,
		Provider:     provider,
		Model:        model,
		Version:      1,
		CreatedAt:    time.Now(),
		FileMetadata: metadata,
		ContentHash:  contentHash,
	}

	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create cache file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(cache); err != nil {
		return fmt.Errorf("failed to encode cache: %w", err)
	}

	fmt.Printf("%sEmbeddings saved to %s (%d chunks, %d files)%s\n",
		ui.ColorGreen, filepath, len(e.Chunks), len(files), ui.ColorReset)
	return nil
}

func (e *Engine) LoadEmbeddings(filepath string) (*EmbeddingCache, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open cache file: %w", err)
	}
	defer file.Close()

	var cache EmbeddingCache
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&cache); err != nil {
		return nil, fmt.Errorf("failed to decode cache: %w", err)
	}

	e.Chunks = cache.Chunks
	fmt.Printf("%sLoaded %d cached embeddings from %s%s\n",
		ui.ColorGreen, len(e.Chunks), filepath, ui.ColorReset)
	fmt.Printf("%s  Pattern: %s | Provider: %s | Model: %s | Created: %s%s\n",
		ui.ColorBlue, cache.GlobPattern, cache.Provider, cache.Model,
		cache.CreatedAt.Format("2006-01-02 15:04"), ui.ColorReset)

	return &cache, nil
}

func (e *Engine) CacheExists(filepath string) bool {
	_, err := os.Stat(filepath)
	return err == nil
}

func GetDefaultCachePath(globPattern string) string {
	safe := strings.ReplaceAll(globPattern, "/", "_")
	safe = strings.ReplaceAll(safe, "*", "all")
	safe = strings.ReplaceAll(safe, ".", "_")

	cacheDir := filepath.Join(os.Getenv("HOME"), ".cache", "ai-rag")
	os.MkdirAll(cacheDir, 0755)

	return filepath.Join(cacheDir, fmt.Sprintf("embeddings_%s.gob", safe))
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

	for i, file := range files {
		content, err := extractText(file)
		if err != nil {
			fmt.Printf("\rSkipping %s: %v", file, err)
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
		fmt.Printf("\rProcessed %d/%d files...", i+1, len(files))
	}
	fmt.Println()

	if len(textsToEmbed) == 0 {
		return fmt.Errorf("no text content extracted")
	}

	fmt.Printf("Generating embeddings for %d chunks...\n", len(textsToEmbed))

	batchSize := 100

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

		progress := float64(end) / float64(len(textsToEmbed)) * 100
		fmt.Printf("\rProgress: %.1f%% (%d/%d chunks)", progress, end, len(textsToEmbed))
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
