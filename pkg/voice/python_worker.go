package voice

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

type PythonWorker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mu     sync.Mutex
}

const pythonScript = `
import sys
import json
import os
import torch
import soundfile as sf
from transformers import pipeline

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    print(json.dumps({"status": "loading"}), flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Defaults
    stt_model = os.getenv("HF_STT_MODEL", "openai/whisper-base.en")
    tts_model = os.getenv("HF_TTS_MODEL", "facebook/mms-tts-eng")

    try:
        # Load STT
        asr = pipeline("automatic-speech-recognition", model=stt_model, device=device)
        
        # Load TTS
        tts = pipeline("text-to-speech", model=tts_model, device=device)
        
        print(json.dumps({"status": "ready", "device": device}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), flush=True)
        return

    for line in sys.stdin:
        try:
            req = json.loads(line)
            req_type = req.get("type")
            
            if req_type == "stt":
                wav_path = req.get("path")
                result = asr(wav_path)
                print(json.dumps({"type": "stt", "text": result.get("text", "").strip()}), flush=True)
                
            elif req_type == "tts":
                text = req.get("text")
                out_path = req.get("out_path")
                
                # Inference
                speech = tts(text)
                
                # Save
                sf.write(out_path, speech["audio"][0], speech["sampling_rate"])
                
                print(json.dumps({"type": "tts", "path": out_path}), flush=True)
                
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
`

func NewPythonWorker(sttModel, ttsModel string) (*PythonWorker, error) {
	// Check for python availability
	pyCmd := "python3"
	if _, err := exec.LookPath(pyCmd); err != nil {
		pyCmd = "python"
		if _, err := exec.LookPath(pyCmd); err != nil {
			return nil, fmt.Errorf("python3/python not found in PATH")
		}
	}

	cmd := exec.Command(pyCmd, "-c", pythonScript)
	cmd.Env = append(os.Environ(),
		"HF_STT_MODEL="+sttModel,
		"HF_TTS_MODEL="+ttsModel,
	)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	// Capture stderr for debugging python crashes
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start python worker: %w", err)
	}

	worker := &PythonWorker{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewScanner(stdoutPipe),
	}

	// Wait for "ready" message
	if worker.stdout.Scan() {
		line := worker.stdout.Bytes()
		var resp map[string]string
		if err := json.Unmarshal(line, &resp); err != nil {
			return nil, fmt.Errorf("bad handshake from python: %s", string(line))
		}
		if resp["status"] == "loading" {
			// Wait for actual ready
			if worker.stdout.Scan() {
				line = worker.stdout.Bytes()
				json.Unmarshal(line, &resp)
			}
		}

		if resp["status"] == "error" {
			return nil, fmt.Errorf("python loading error: %s", resp["message"])
		}
		if resp["status"] != "ready" {
			return nil, fmt.Errorf("unexpected python status: %s", string(line))
		}
	} else {
		return nil, fmt.Errorf("python worker exited immediately")
	}

	return worker, nil
}

func (w *PythonWorker) Close() {
	w.stdin.Close()
	if w.cmd.Process != nil {
		w.cmd.Process.Kill()
	}
}

func (w *PythonWorker) STT(wavPath string) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	req := map[string]string{
		"type": "stt",
		"path": wavPath,
	}
	bytes, _ := json.Marshal(req)
	fmt.Fprintln(w.stdin, string(bytes))

	if w.stdout.Scan() {
		var resp struct {
			Text  string `json:"text"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(w.stdout.Bytes(), &resp); err != nil {
			return "", err
		}
		if resp.Error != "" {
			return "", fmt.Errorf(resp.Error)
		}
		return resp.Text, nil
	}
	return "", fmt.Errorf("python worker closed stream")
}

func (w *PythonWorker) TTS(text string, outPath string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	req := map[string]string{
		"type":     "tts",
		"text":     text,
		"out_path": outPath,
	}
	bytes, _ := json.Marshal(req)
	fmt.Fprintln(w.stdin, string(bytes))

	if w.stdout.Scan() {
		var resp struct {
			Path  string `json:"path"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal(w.stdout.Bytes(), &resp); err != nil {
			return err
		}
		if resp.Error != "" {
			return fmt.Errorf(resp.Error)
		}
		return nil
	}
	return fmt.Errorf("python worker closed stream")
}
