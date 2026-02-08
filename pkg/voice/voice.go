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
	"time"

	"github.com/gordonklaus/portaudio"
	openai "github.com/sashabaranov/go-openai"
)

type Manager struct {
	client *openai.Client
}

func NewManager(apiKey string) (*Manager, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key required for voice")
	}
	if err := portaudio.Initialize(); err != nil {
		return nil, fmt.Errorf("portaudio init error: %w", err)
	}
	return &Manager{
		client: openai.NewClient(apiKey),
	}, nil
}

func (m *Manager) Close() {
	portaudio.Terminate()
}

func (m *Manager) RecordUntilSpace(inputReader interface {
	ReadRune() (rune, int, error)
}) ([]byte, error) {
	const sampleRate = 44100
	const channels = 1

	var buffer []int16

	stream, err := portaudio.OpenDefaultStream(channels, 0, sampleRate, 0, func(in []int16) {
		buffer = append(buffer, in...)
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

func (m *Manager) Speak(ctx context.Context, text string) error {
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
