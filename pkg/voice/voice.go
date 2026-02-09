package voice

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/gordonklaus/portaudio"
	openai "github.com/sashabaranov/go-openai"
	"github.com/yuriiter/ai/pkg/config"
	"github.com/yuriiter/ai/pkg/ui"
)

type Manager struct {
	client   *openai.Client
	config   config.Config
	pyWorker *PythonWorker
}

func NewManager(cfg config.Config) (*Manager, error) {
	// If using local HF, we need to spin up the python worker
	var worker *PythonWorker
	var err error

	if cfg.VoiceProvider == "local-hf" {
		fmt.Printf("%sInitializing Python Local HF Worker (this may take a moment to load models)...%s\n", ui.ColorBlue, ui.ColorReset)
		fmt.Printf("  STT Model: %s\n  TTS Model: %s\n", cfg.HF_STT_Model, cfg.HF_TTS_Model)

		worker, err = NewPythonWorker(cfg.HF_STT_Model, cfg.HF_TTS_Model)
		if err != nil {
			return nil, fmt.Errorf("failed to init python worker: %w\nEnsure you have installed: pip install torch transformers scipy soundfile", err)
		}
		fmt.Printf("%sModels loaded successfully.%s\n", ui.ColorGreen, ui.ColorReset)
	} else if cfg.ApiKey == "" {
		return nil, fmt.Errorf("API key required for cloud voice")
	}

	suppressStderr(func() {
		portaudio.Initialize()
	})

	return &Manager{
		client:   openai.NewClient(cfg.ApiKey),
		config:   cfg,
		pyWorker: worker,
	}, nil
}

func (m *Manager) Close() {
	if m.pyWorker != nil {
		m.pyWorker.Close()
	}
	portaudio.Terminate()
}

func (m *Manager) RecordUntilSpace(inputReader interface {
	ReadRune() (rune, int, error)
}) ([]byte, error) {
	const sampleRate = 44100
	const channels = 1

	var buffer []int16

	var stream *portaudio.Stream
	var err error

	suppressStderr(func() {
		stream, err = portaudio.OpenDefaultStream(channels, 0, sampleRate, 0, func(in []int16) {
			buffer = append(buffer, in...)
		})
	})
	if err != nil {
		return nil, err
	}

	if err := stream.Start(); err != nil {
		return nil, err
	}

	for {
		r, _, err := inputReader.ReadRune()
		if err != nil {
			break
		}
		if r == ' ' {
			break
		}
	}

	if err := stream.Stop(); err != nil {
		return nil, err
	}
	stream.Close()

	return encodeWAV(buffer, sampleRate), nil
}

func (m *Manager) Transcribe(ctx context.Context, wavData []byte) (string, error) {
	if m.config.VoiceProvider == "local-hf" {
		return m.transcribeHF(wavData)
	}

	req := openai.AudioRequest{
		Model:    openai.Whisper1,
		Reader:   bytes.NewReader(wavData),
		FilePath: "voice.wav",
	}
	resp, err := m.client.CreateTranscription(ctx, req)
	if err != nil {
		return "", err
	}
	return resp.Text, nil
}

func (m *Manager) transcribeHF(wavData []byte) (string, error) {
	tmpDir := os.TempDir()
	wavFile := filepath.Join(tmpDir, fmt.Sprintf("rec_%d.wav", time.Now().UnixNano()))

	if err := os.WriteFile(wavFile, wavData, 0644); err != nil {
		return "", err
	}
	defer os.Remove(wavFile)

	return m.pyWorker.STT(wavFile)
}

func (m *Manager) Speak(ctx context.Context, text string) error {
	if m.config.VoiceProvider == "local-hf" {
		return m.speakHF(text)
	}

	req := openai.CreateSpeechRequest{
		Model:          openai.TTSModel1,
		Input:          text,
		Voice:          openai.VoiceAlloy,
		ResponseFormat: openai.SpeechResponseFormatMp3,
	}

	resp, err := m.client.CreateSpeech(ctx, req)
	if err != nil {
		return err
	}
	defer resp.Close()

	tmpDir := os.TempDir()
	tmpFile := filepath.Join(tmpDir, fmt.Sprintf("ai_speech_%d.mp3", time.Now().UnixNano()))

	f, err := os.Create(tmpFile)
	if err != nil {
		return err
	}

	if _, err := io.Copy(f, resp); err != nil {
		f.Close()
		return err
	}
	f.Close()

	return playAudioFile(tmpFile)
}

func (m *Manager) speakHF(text string) error {
	tmpDir := os.TempDir()
	outFile := filepath.Join(tmpDir, fmt.Sprintf("ai_hf_speech_%d.wav", time.Now().UnixNano()))

	err := m.pyWorker.TTS(text, outFile)
	if err != nil {
		return err
	}
	// defer os.Remove(outFile) // Optional: keep for debug or remove

	return playAudioFile(outFile)
}

func encodeWAV(data []int16, sampleRate int) []byte {
	buf := new(bytes.Buffer)

	dataSize := len(data) * 2
	totalSize := dataSize + 36

	buf.Write([]byte("RIFF"))
	binary.Write(buf, binary.LittleEndian, int32(totalSize))
	buf.Write([]byte("WAVE"))
	buf.Write([]byte("fmt "))
	binary.Write(buf, binary.LittleEndian, int32(16))
	binary.Write(buf, binary.LittleEndian, int16(1))
	binary.Write(buf, binary.LittleEndian, int16(1))
	binary.Write(buf, binary.LittleEndian, int32(sampleRate))
	binary.Write(buf, binary.LittleEndian, int32(sampleRate*2))
	binary.Write(buf, binary.LittleEndian, int16(2))
	binary.Write(buf, binary.LittleEndian, int16(16))

	buf.Write([]byte("data"))
	binary.Write(buf, binary.LittleEndian, int32(dataSize))

	binary.Write(buf, binary.LittleEndian, data)

	return buf.Bytes()
}

func playAudioFile(path string) error {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("afplay", path)
	case "linux":
		if _, err := exec.LookPath("mpg123"); err == nil {
			cmd = exec.Command("mpg123", path)
		} else if _, err := exec.LookPath("ffplay"); err == nil {
			cmd = exec.Command("ffplay", "-nodisp", "-autoexit", path)
		} else if _, err := exec.LookPath("aplay"); err == nil {
			cmd = exec.Command("aplay", path)
		} else {
			return fmt.Errorf("no audio player found (install mpg123 or ffmpeg)")
		}
	case "windows":
		cmd = exec.Command("powershell", "-c", fmt.Sprintf("(New-Object Media.SoundPlayer '%s').PlaySync();", path))
	default:
		return fmt.Errorf("unsupported OS for playback")
	}

	return cmd.Run()
}

func suppressStderr(f func()) {
	null, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0666)
	if err != nil {
		f()
		return
	}
	defer null.Close()

	stderrFd := int(os.Stderr.Fd())
	originalStderr, err := syscall.Dup(stderrFd)
	if err != nil {
		f()
		return
	}

	syscall.Dup2(int(null.Fd()), stderrFd)
	f()
	syscall.Dup2(originalStderr, stderrFd)
	syscall.Close(originalStderr)
}
