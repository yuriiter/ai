package main

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

//go:embed model_runner.py
var modelRunnerPy []byte

type AIRequest struct {
	Task   string         `json:"task"`
	Model  string         `json:"model"`
	Input  string         `json:"input"`
	Params map[string]any `json:"params"`
}

type AIResponse struct {
	Status     string `json:"status"`
	Text       string `json:"text"`
	File       string `json:"file"`
	Language   string `json:"language"`
	Error      string `json:"error"`
	Device     string `json:"device"`
	Model      string `json:"model"`
	SampleRate int    `json:"sample_rate"`
}

type Worker struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	scanner *bufio.Scanner
	tmpPy   string
}

func NewWorker(python string) (*Worker, error) {
	tmpFile, err := os.CreateTemp("", "model_runner_*.py")
	if err != nil {
		return nil, fmt.Errorf("create temp script: %w", err)
	}
	if _, err := tmpFile.Write(modelRunnerPy); err != nil {
		return nil, fmt.Errorf("write temp script: %w", err)
	}
	tmpFile.Close()

	cmd := exec.Command(python, tmpFile.Name())
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start python: %w", err)
	}

	w := &Worker{
		cmd:     cmd,
		stdin:   stdin,
		scanner: bufio.NewScanner(stdout),
		tmpPy:   tmpFile.Name(),
	}

	if err := w.readReady(); err != nil {
		_ = cmd.Process.Kill()
		return nil, err
	}

	return w, nil
}

func (w *Worker) readReady() error {
	if !w.scanner.Scan() {
		return errors.New("python worker did not send ready signal")
	}
	var resp AIResponse
	if err := json.Unmarshal(w.scanner.Bytes(), &resp); err != nil {
		return fmt.Errorf("bad ready payload: %w", err)
	}
	if resp.Status != "ready" {
		return fmt.Errorf("unexpected status from worker: %s", resp.Status)
	}
	fmt.Printf("Python worker ready  device=%s\n", resp.Device)
	return nil
}

func (w *Worker) Call(req AIRequest) (AIResponse, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := json.Marshal(req)
	if err != nil {
		return AIResponse{}, fmt.Errorf("marshal request: %w", err)
	}
	data = append(data, '\n')

	if _, err := w.stdin.Write(data); err != nil {
		return AIResponse{}, fmt.Errorf("write to worker stdin: %w", err)
	}

	if !w.scanner.Scan() {
		return AIResponse{}, errors.New("worker closed stdout unexpectedly")
	}

	var resp AIResponse
	if err := json.Unmarshal(w.scanner.Bytes(), &resp); err != nil {
		return AIResponse{}, fmt.Errorf("unmarshal response: %w", err)
	}
	return resp, nil
}

func (w *Worker) Close() {
	_ = w.stdin.Close()
	_ = w.cmd.Wait()
	_ = os.Remove(w.tmpPy)
}

func Transcribe(w *Worker, model, audio, lang string) (string, string, error) {
	params := map[string]any{}
	if lang != "" {
		params["language"] = lang
	}

	resp, err := w.Call(AIRequest{
		Task:   "stt",
		Model:  model,
		Input:  audio,
		Params: params,
	})
	if err != nil {
		return "", "", err
	}
	if resp.Status != "success" {
		return "", "", fmt.Errorf("stt error: %s", resp.Error)
	}
	return resp.Text, resp.Model, nil
}

func Synthesize(w *Worker, model, text, outputPath string, params map[string]any) (string, error) {
	if params == nil {
		params = map[string]any{}
	}
	if outputPath != "" {
		params["output_path"] = outputPath
	}

	resp, err := w.Call(AIRequest{
		Task:   "tts",
		Model:  model,
		Input:  text,
		Params: params,
	})
	if err != nil {
		return "", err
	}
	if resp.Status != "success" {
		return "", fmt.Errorf("tts error: %s", resp.Error)
	}
	return resp.File, nil
}

func printUsage() {
	fmt.Println(`
ai_voice — fast TTS / STT using tiny HuggingFace models

USAGE
  ai_voice stt  --audio <file>  [--model openai/whisper-tiny] [--lang en]
  ai_voice tts  --text  <text>  [--model facebook/mms-tts-eng] [--out output.wav]
  ai_voice info
  ai_voice dump

SUBCOMMANDS
  stt   Transcribe audio to text
  tts   Synthesise text to WAV
  info  Show device and loaded models
  dump  Extract embedded model_runner.py to current directory

STT OPTIONS
  --audio       Path to audio file (WAV / MP3 / FLAC / M4A)
  --model       openai/whisper-tiny   ~39 MB  (default)
                openai/whisper-base   ~145 MB
                openai/whisper-small  ~244 MB
  --lang        Language code (en, de, fr …); omit for auto-detect
  --timestamps  Include word-level timestamps

TTS OPTIONS
  --text        Text to synthesise
  --model       facebook/mms-tts-eng  ~50 MB  (default, English)
                facebook/mms-tts-deu  ~50 MB  (German)
                facebook/mms-tts-fra  ~50 MB  (French)
                microsoft/speecht5_tts ~308 MB (higher quality)
  --out         Output WAV path (default: temp file)
  --speaker     SpeechT5 speaker index 0-7508 (default 7306)

PYTHON
  --python      Python interpreter (default: python3)
`)
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	pythonFlag := flag.String("python", "python3", "Python interpreter")
	flag.CommandLine.Parse([]string{})

	subcommand := os.Args[1]

	switch subcommand {
	case "dump":
		dst := "model_runner.py"
		if err := os.WriteFile(dst, modelRunnerPy, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "dump error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Wrote %d bytes → %s\n", len(modelRunnerPy), dst)

	case "stt":
		fs := flag.NewFlagSet("stt", flag.ExitOnError)
		audio := fs.String("audio", "", "path to audio file (required)")
		model := fs.String("model", "openai/whisper-tiny", "HuggingFace ASR model id")
		lang := fs.String("lang", "", "language code or '' for auto-detect")
		timestamps := fs.Bool("timestamps", false, "return word-level timestamps")
		python := fs.String("python", *pythonFlag, "Python interpreter")
		_ = fs.Parse(os.Args[2:])

		if *audio == "" {
			fmt.Fprintln(os.Stderr, "error: --audio is required")
			fs.Usage()
			os.Exit(1)
		}
		if !filepath.IsAbs(*audio) {
			abs, _ := filepath.Abs(*audio)
			*audio = abs
		}

		fmt.Println("Starting Python worker…")
		w, err := NewWorker(*python)
		if err != nil {
			fatalf("worker init: %v\n(pip install transformers torch soundfile)\n", err)
		}
		defer w.Close()

		fmt.Printf("Transcribing %s  model=%s …\n", *audio, *model)
		params := map[string]any{}
		if *lang != "" {
			params["language"] = *lang
		}
		if *timestamps {
			params["return_timestamps"] = true
		}
		resp, err := w.Call(AIRequest{Task: "stt", Model: *model, Input: *audio, Params: params})
		if err != nil {
			fatalf("transcription failed: %v\n", err)
		}
		if resp.Status != "success" {
			fatalf("stt error: %s\n", resp.Error)
		}
		fmt.Printf("\n─────────────────────────────────\n")
		fmt.Printf("Model    : %s\n", resp.Model)
		fmt.Printf("Text     : %s\n", resp.Text)
		fmt.Printf("─────────────────────────────────\n")

	case "tts":
		fs := flag.NewFlagSet("tts", flag.ExitOnError)
		text := fs.String("text", "", "text to synthesise (required)")
		model := fs.String("model", "facebook/mms-tts-eng", "HuggingFace TTS model id")
		out := fs.String("out", "", "output WAV path (default: temp file)")
		speaker := fs.Int("speaker", -1, "SpeechT5 speaker index 0-7508 (-1 = default)")
		python := fs.String("python", *pythonFlag, "Python interpreter")
		_ = fs.Parse(os.Args[2:])

		if *text == "" && len(fs.Args()) > 0 {
			*text = strings.Join(fs.Args(), " ")
		}
		if *text == "" {
			fmt.Fprintln(os.Stderr, "error: --text is required")
			fs.Usage()
			os.Exit(1)
		}

		fmt.Println("Starting Python worker…")
		w, err := NewWorker(*python)
		if err != nil {
			fatalf("worker init: %v\n(pip install transformers torch soundfile datasets)\n", err)
		}
		defer w.Close()

		params := map[string]any{}
		if *out != "" {
			params["output_path"] = *out
		}
		if *speaker >= 0 {
			params["speaker_idx"] = *speaker
		}
		fmt.Printf("Synthesising with model=%s …\n", *model)
		resp, err := w.Call(AIRequest{Task: "tts", Model: *model, Input: *text, Params: params})
		if err != nil {
			fatalf("synthesis failed: %v\n", err)
		}
		if resp.Status != "success" {
			fatalf("tts error: %s\n", resp.Error)
		}
		fmt.Printf("\n─────────────────────────────────\n")
		fmt.Printf("Model      : %s\n", resp.Model)
		fmt.Printf("Sample rate: %d Hz\n", resp.SampleRate)
		fmt.Printf("WAV output : %s\n", resp.File)
		fmt.Printf("─────────────────────────────────\n")

	case "info":
		python := pythonFlag
		if len(os.Args) > 2 {
			for i, a := range os.Args[2:] {
				if a == "--python" && i+1 < len(os.Args[2:]) {
					*python = os.Args[3+i]
				}
			}
		}
		w, err := NewWorker(*python)
		if err != nil {
			fatalf("worker init: %v\n", err)
		}
		defer w.Close()
		resp, err := w.Call(AIRequest{Task: "info"})
		if err != nil {
			fatalf("info call failed: %v\n", err)
		}
		data, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(data))

	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand: %q\n", subcommand)
		printUsage()
		os.Exit(1)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format, args...)
	os.Exit(1)
}
