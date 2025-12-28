package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
	ID      int         `json:"id,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
	ID int `json:"id"`
}

type Client struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    *bufio.Scanner
	idCounter int
	mu        sync.Mutex
}

func NewClient(command string) (*Client, error) {
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty command")
	}

	cmd := exec.Command(parts[0], parts[1:]...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(stdoutPipe)
	buf := make([]byte, 1024*1024*2)
	scanner.Buffer(buf, 1024*1024*2)

	client := &Client{
		cmd:       cmd,
		stdin:     stdin,
		stdout:    scanner,
		idCounter: 0,
	}

	return client, client.initialize()
}

func (c *Client) initialize() error {
	initParams := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities": map[string]interface{}{
			"tools": map[string]interface{}{},
		},
		"clientInfo": map[string]string{
			"name":    "go-cli-ai",
			"version": "1.0.0",
		},
	}

	_, err := c.Call("initialize", initParams)
	if err != nil {
		return fmt.Errorf("mcp handshake failed: %w", err)
	}

	c.notify("notifications/initialized", nil)
	return nil
}

func (c *Client) Call(method string, params interface{}) (json.RawMessage, error) {
	c.mu.Lock()
	c.idCounter++
	id := c.idCounter
	c.mu.Unlock()

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      id,
	}

	bytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	if _, err := c.stdin.Write(append(bytes, '\n')); err != nil {
		return nil, err
	}

	for c.stdout.Scan() {
		line := c.stdout.Bytes()

		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			continue
		}

		if resp.ID == id {
			if resp.Error != nil {
				return nil, fmt.Errorf("server error code %d: %s", resp.Error.Code, resp.Error.Message)
			}
			return resp.Result, nil
		}
	}

	if err := c.stdout.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("connection closed or response not received")
}

func (c *Client) notify(method string, params interface{}) {
	req := JSONRPCRequest{JSONRPC: "2.0", Method: method, Params: params}
	bytes, _ := json.Marshal(req)
	c.stdin.Write(append(bytes, '\n'))
}

func (c *Client) Close() {
	c.stdin.Close()
	if c.cmd != nil && c.cmd.Process != nil {
		c.cmd.Process.Kill()
	}
}
