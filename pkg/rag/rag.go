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
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ledongthuc/pdf"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/taylorskalyo/goreader/epub"
	"github.com/yuriiter/ai/pkg/ui"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type LocalEmbedder struct {
	interfaceModel textencoding.Interface
	mu             sync.Mutex
}

func NewLocalEmbedder() (*LocalEmbedder, error) {
	fmt.Printf("%sInitializing local embedding model (downloading if needed)...%s\n", ui.ColorBlue, ui.ColorReset)

	zerolog.SetGlobalLevel(zerolog.WarnLevel)

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

	numWorkers := runtime.NumCPU()
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
				vec, err := l.safeEncode(ctx, j.text)
				if err != nil {
					fmt.Printf("\nWarning: Skipping chunk %d due to encoding error: %v\n", j.index, err)
					continue
				}

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

	return results, nil
}

func (l *LocalEmbedder) safeEncode(ctx context.Context, text string) ([]float32, error) {
	output, err := l.interfaceModel.Encode(ctx, text, int(bert.MeanPooling))
	if err == nil {
		return output.Vector.Data().F32(), nil
	}

	if len(text) > 1024 {
		safeText := text[:1024]
		output, err = l.interfaceModel.Encode(ctx, safeText, int(bert.MeanPooling))
		if err == nil {
			return output.Vector.Data().F32(), nil
		}
	}

	if len(text) > 512 {
		safeText := text[:512]
		output, err = l.interfaceModel.Encode(ctx, safeText, int(bert.MeanPooling))
		if err == nil {
			return output.Vector.Data().F32(), nil
		}
	}

	return nil, err
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
	GlobPatterns []string
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

func New() (*Engine, error) {
	emb, err := NewLocalEmbedder()
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

func (e *Engine) ValidateCache(cachePath string, globPatterns []string) (bool, string) {
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

	if len(cache.GlobPatterns) != len(globPatterns) {
		return false, "pattern count mismatch"
	}

	sort.Strings(cache.GlobPatterns)
	currentPatterns := make([]string, len(globPatterns))
	copy(currentPatterns, globPatterns)
	sort.Strings(currentPatterns)

	for i := range cache.GlobPatterns {
		if cache.GlobPatterns[i] != currentPatterns[i] {
			return false, "pattern mismatch"
		}
	}

	currentFiles := FindFiles(globPatterns)
	if len(currentFiles) == 0 {
		return false, "no files found matching patterns"
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
			return false, fmt.Sprintf("file changed: %s", current.Path)
		}
	}

	return true, ""
}

func (e *Engine) SaveEmbeddings(filepath string, globPatterns []string) error {
	files := FindFiles(globPatterns)
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
		GlobPatterns: globPatterns,
		Provider:     "local",
		Model:        "sentence-transformers/all-MiniLM-L6-v2",
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
	fmt.Printf("%s  Patterns: %s | Provider: %s | Model: %s | Created: %s%s\n",
		ui.ColorBlue, strings.Join(cache.GlobPatterns, ", "), cache.Provider, cache.Model,
		cache.CreatedAt.Format("2006-01-02 15:04"), ui.ColorReset)

	return &cache, nil
}

func (e *Engine) CacheExists(filepath string) bool {
	_, err := os.Stat(filepath)
	return err == nil
}

func GetDefaultCachePath(globPatterns []string) string {
	sort.Strings(globPatterns)

	cwd, err := os.Getwd()
	if err != nil {
		cwd = "unknown"
	}

	combined := fmt.Sprintf("%s:%s", cwd, strings.Join(globPatterns, ";"))

	hasher := sha256.New()
	hasher.Write([]byte(combined))
	hash := hex.EncodeToString(hasher.Sum(nil))[:16]

	cacheDir := filepath.Join(os.Getenv("HOME"), ".cache", "ai-rag")
	os.MkdirAll(cacheDir, 0755)

	return filepath.Join(cacheDir, fmt.Sprintf("rag_%s.gob", hash))
}

func (e *Engine) IngestGlobs(ctx context.Context, globPatterns []string) error {
	files := FindFiles(globPatterns)
	if len(files) == 0 {
		return fmt.Errorf("no files found matching patterns")
	}

	fmt.Printf("%sRAG: Found %d files. Processing...%s\n", ui.ColorBlue, len(files), ui.ColorReset)

	var textsToEmbed []string
	var mapIndexToMeta []struct {
		Text     string
		Filename string
	}

	for i, file := range files {
		content, err := ExtractText(file)
		if err != nil {
			fmt.Printf("\rSkipping %s: %v", file, err)
			continue
		}

		content = cleanText(content)

		if content == "" {
			continue
		}

		chunks := chunkText(content, 800, 100)
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
			if len(vec) == 0 {
				continue
			}

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
	if len(vectors) == 0 || len(vectors[0]) == 0 {
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

func FindFiles(patterns []string) []string {
	var files []string
	seen := make(map[string]bool)

	var expandedPatterns []string
	for _, p := range patterns {
		if s := strings.Index(p, "{"); s != -1 {
			if e := strings.LastIndex(p, "}"); e != -1 && e > s {
				prefix := p[:s]
				suffix := p[e+1:]
				opts := strings.Split(p[s+1:e], ",")
				for _, o := range opts {
					expandedPatterns = append(expandedPatterns, prefix+strings.TrimSpace(o)+suffix)
				}
				continue
			}
		}
		expandedPatterns = append(expandedPatterns, p)
	}

	for _, pattern := range expandedPatterns {
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
						if !seen[path] {
							files = append(files, path)
							seen[path] = true
						}
					}
				}
				return nil
			})
		} else {
			f, _ := filepath.Glob(pattern)
			for _, file := range f {
				if !seen[file] {
					files = append(files, file)
					seen[file] = true
				}
			}
		}
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

func cleanText(s string) string {
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.ReplaceAll(s, "\r", " ")

	reNewlines := regexp.MustCompile(`\n{2,}`)
	s = reNewlines.ReplaceAllString(s, "\n")

	reSpaces := regexp.MustCompile(` {2,}`)
	s = reSpaces.ReplaceAllString(s, " ")

	s = strings.Map(func(r rune) rune {
		if r < 32 && r != '\n' {
			return -1
		}
		return r
	}, s)

	return strings.TrimSpace(s)
}

func ExtractText(path string) (text string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic recovering file %s: %v", path, r)
		}
	}()

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
		total := r.NumPage()
		for i := 1; i <= total; i++ {
			p := r.Page(i)
			if !p.V.IsNull() {
				t, err := p.GetPlainText(nil)
				if err != nil {
					continue
				}
				sb.WriteString(t)
				sb.WriteString("\n")
			}
		}
		return sb.String(), nil
	case ".docx":
		return parseDocx(path)
	case ".xlsx":
		return parseXlsx(path)
	case ".epub":
		rc, err := epub.OpenReader(path)
		if err != nil {
			return "", err
		}
		defer rc.Close()
		var sb strings.Builder
		if len(rc.Rootfiles) > 0 {
			for _, item := range rc.Rootfiles[0].Manifest.Items {
				if strings.Contains(item.MediaType, "html") {
					f, err := item.Open()
					if err != nil {
						continue
					}
					b, _ := io.ReadAll(f)
					f.Close()
					sb.WriteString(stripTags(string(b)) + "\n")
				}
			}
		}
		return sb.String(), nil
	case ".fb2":
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return stripTags(string(b)), nil
	}
	return "", fmt.Errorf("unsupported type: %s", ext)
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
			rc, err := f.Open()
			if err != nil {
				return "", err
			}
			defer rc.Close()
			dec := xml.NewDecoder(rc)
			for {
				t, _ := dec.Token()
				if t == nil {
					break
				}
				if se, ok := t.(xml.StartElement); ok && (se.Name.Local == "t") {
					var s string
					dec.DecodeElement(&s, &se)
					sb.WriteString(s)
					sb.WriteString(" ")
				}
				if se, ok := t.(xml.StartElement); ok && (se.Name.Local == "p" || se.Name.Local == "br") {
					sb.WriteString("\n")
				}
			}
		}
	}
	return sb.String(), nil
}

func parseXlsx(path string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	var sb strings.Builder
	for _, f := range r.File {
		if f.Name == "xl/sharedStrings.xml" {
			rc, err := f.Open()
			if err != nil {
				return "", err
			}
			defer rc.Close()
			dec := xml.NewDecoder(rc)
			for {
				t, _ := dec.Token()
				if t == nil {
					break
				}
				if se, ok := t.(xml.StartElement); ok && (se.Name.Local == "t") {
					var s string
					dec.DecodeElement(&s, &se)
					sb.WriteString(s)
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
