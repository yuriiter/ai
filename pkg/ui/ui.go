package ui

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

var (
	ColorRed   = "\033[31m"
	ColorGreen = "\033[32m"
	ColorBlue  = "\033[34m"
	ColorReset = "\033[0m"
)

func init() {
	if !IsStdoutTTY() {
		ColorRed, ColorGreen, ColorBlue, ColorReset = "", "", "", ""
	}
}

func IsStdoutTTY() bool {
	stat, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}

func IsStdinPiped() bool {
	stat, _ := os.Stdin.Stat()
	return (stat.Mode() & os.ModeCharDevice) == 0
}

func GatherInput(args []string, useEditor bool, editorCmd string) (string, error) {
	var initialContent string
	if len(args) > 0 {
		initialContent = strings.Join(args, " ")
	}

	if IsStdinPiped() {
		stdinBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", err
		}
		if initialContent != "" {
			initialContent = fmt.Sprintf("%s\n\n---\n%s", initialContent, string(stdinBytes))
		} else {
			initialContent = string(stdinBytes)
		}
	}

	if useEditor {
		return OpenEditor(editorCmd, initialContent)
	}

	return initialContent, nil
}

func OpenEditor(editor string, content string) (string, error) {
	tmpFile, err := os.CreateTemp("", "ai-prompt-*.md")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmpFile.Name())

	if content != "" {
		tmpFile.WriteString(content)
	}
	tmpFile.Close()

	cmd := exec.Command(editor, tmpFile.Name())
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to run editor %q: %w", editor, err)
	}

	finalBytes, err := os.ReadFile(tmpFile.Name())
	if err != nil {
		return "", err
	}
	return string(finalBytes), nil
}

func PrintUserPrompt(prompt string) {
	fmt.Printf("%s> %s%s\n", ColorBlue, prompt, ColorReset)
}

func PrintAgentMessage(msg string) {
	fmt.Printf("%s%s%s", ColorGreen, msg, ColorReset)
}

func PrintToolUse(toolName string, args string) {
	fmt.Printf("%s[Agent using tool: %s (%s)]%s\n", ColorRed, toolName, args, ColorReset)
}
