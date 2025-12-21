package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"github.com/spf13/cobra"
)

type Config struct {
	ApiKey             string
	BaseURL            string
	Model              string
	Editor             string
	SystemInstructions string
}

var (
	ColorRed   = "\033[31m"
	ColorGreen = "\033[32m"
	ColorReset = "\033[0m"
)

func isStdoutTTY() bool {
	stat, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}

func main() {
	if !isStdoutTTY() {
		ColorRed = ""
		ColorGreen = ""
		ColorReset = ""
	}

	var editorFlag bool
	var streamingFlag bool

	var rootCmd = &cobra.Command{
		Use:   "ai [prompt...]",
		Short: "A CLI tool for interacting with OpenAI-compatible APIs",
		Run: func(cmd *cobra.Command, args []string) {
			config := loadConfig()
			if config.ApiKey == "" {
				fmt.Fprintf(os.Stderr, "%sError: AI_API_KEY environment variable is not set.%s\n", ColorRed, ColorReset)
				os.Exit(1)
			}

			prompt, err := gatherInput(args, editorFlag, config.Editor)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%sError processing input: %v%s\n", ColorRed, err, ColorReset)
				os.Exit(1)
			}

			if strings.TrimSpace(prompt) == "" {
				cmd.Help()
				os.Exit(0)
			}

			if err := runCompletion(config, prompt, streamingFlag); err != nil {
				fmt.Fprintf(os.Stderr, "\n%sAPI Error: %v%s\n", ColorRed, err, ColorReset)
				os.Exit(1)
			}
		},
	}

	rootCmd.Flags().BoolVarP(&editorFlag, "editor", "e", false, "Open editor to compose prompt")
	rootCmd.Flags().BoolVarP(&streamingFlag, "streaming", "s", false, "Use streaming output")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "%s%v%s\n", ColorRed, err, ColorReset)
		os.Exit(1)
	}
}

func loadConfig() Config {
	c := Config{
		ApiKey:             os.Getenv("AI_API_KEY"),
		BaseURL:            os.Getenv("AI_BASE_URL"),
		Model:              os.Getenv("AI_MODEL"),
		Editor:             os.Getenv("EDITOR"),
		SystemInstructions: os.Getenv("AI_SYSTEM_INSTRUCTIONS"),
	}

	if c.Model == "" {
		c.Model = "gpt-3.5-turbo"
	}
	if c.Editor == "" {
		if _, err := exec.LookPath("vim"); err == nil {
			c.Editor = "vim"
		} else if _, err := exec.LookPath("nano"); err == nil {
			c.Editor = "nano"
		} else {
			c.Editor = "vi"
		}
	}

	return c
}

func gatherInput(args []string, useEditor bool, editorCmd string) (string, error) {
	var initialContent string

	if len(args) > 0 {
		initialContent = strings.Join(args, " ")
	}

	stat, _ := os.Stdin.Stat()
	isPiped := (stat.Mode() & os.ModeCharDevice) == 0

	if isPiped {
		stdinBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", err
		}
		stdinContent := string(stdinBytes)

		if initialContent != "" {
			initialContent = fmt.Sprintf("%s\n\n---\n%s", initialContent, stdinContent)
		} else {
			initialContent = stdinContent
		}
	}

	if useEditor {
		editorOutput, err := openEditor(editorCmd)

		if err != nil {
			return "", err
		}

		if strings.TrimSpace(initialContent) != "" && strings.TrimSpace(editorOutput) != "" {
			initialContent = fmt.Sprintf("%s\n\n---\n%s", initialContent, editorOutput)
		} else if strings.TrimSpace(editorOutput) != "" {
			initialContent = editorOutput
		}
	}

	return initialContent, nil
}

func openEditor(editor string) (string, error) {
	tmpFile, err := os.CreateTemp("", "ai-prompt-*.md")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmpFile.Name())

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

func runCompletion(cfg Config, prompt string, isStreaming bool) error {
	config := openai.DefaultConfig(cfg.ApiKey)

	if cfg.BaseURL != "" {
		config.BaseURL = cfg.BaseURL
	}

	client := openai.NewClientWithConfig(config)
	ctx := context.Background()

	messages := make([]openai.ChatCompletionMessage, 0)

	if strings.TrimSpace(cfg.SystemInstructions) != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: cfg.SystemInstructions,
		})
	}

	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	req := openai.ChatCompletionRequest{
		Model:    cfg.Model,
		Messages: messages,
		Stream:   isStreaming,
	}

	if isStreaming {
		stream, err := client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			return err
		}
		defer stream.Close()

		fmt.Print(ColorGreen)

		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				fmt.Print(ColorReset)
				fmt.Println()
				return nil
			}
			if err != nil {
				fmt.Print(ColorReset)
				return err
			}

			fmt.Print(response.Choices[0].Delta.Content)
		}
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return err
	}

	fmt.Print(ColorGreen)
	if len(resp.Choices) > 0 {
		fmt.Print(resp.Choices[0].Message.Content)
	}
	fmt.Println(ColorReset)

	return nil
}
