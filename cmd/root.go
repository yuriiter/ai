package cmd

import (
	"bufio"
	"context"
	"fmt"
	"github.com/yuriiter/ai/pkg/agent"
	"github.com/yuriiter/ai/pkg/config"
	"github.com/yuriiter/ai/pkg/ui"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

var (
	editorFlag      bool
	interactiveFlag bool
	agentFlag       bool
	memoryFlag      bool
	stepsFlag       int
	temperatureFlag float32
	mcpFlags        []string
)

var rootCmd = &cobra.Command{
	Use:   "ai [prompt...]",
	Short: "A CLI AI Agent with optional MCP support",
	Run: func(cmd *cobra.Command, args []string) {
		cfg := config.Load()
		if cfg.ApiKey == "" {
			fmt.Fprintf(os.Stderr, "%sError: AI_API_KEY not set.%s\n", ui.ColorRed, ui.ColorReset)
			os.Exit(1)
		}

		cfg.MaxSteps = stepsFlag
		cfg.RetainHistory = memoryFlag
		cfg.Temperature = temperatureFlag

		aiAgent, err := agent.New(cfg, agentFlag, mcpFlags)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sError initializing agent: %v%s\n", ui.ColorRed, err, ui.ColorReset)
			os.Exit(1)
		}
		defer aiAgent.Close()

		ctx := context.Background()

		if interactiveFlag {
			startInteractive(ctx, aiAgent)
			return
		}

		prompt, err := ui.GatherInput(args, editorFlag, cfg.Editor)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
			os.Exit(1)
		}

		if strings.TrimSpace(prompt) == "" {
			cmd.Help()
			os.Exit(0)
		}

		if err := aiAgent.RunTurn(ctx, prompt, true); err != nil {
			fmt.Fprintf(os.Stderr, "\nAPI Error: %v\n", err)
			os.Exit(1)
		}
	},
}

func startInteractive(ctx context.Context, ai *agent.Agent) {
	fmt.Println("Interactive Mode. Type 'exit' to quit.")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("\n%s>> %s", ui.ColorBlue, ui.ColorReset)
		if !scanner.Scan() {
			break
		}
		text := scanner.Text()
		if text == "exit" || text == "quit" {
			break
		}
		if err := ai.RunTurn(ctx, text, true); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	}
}

func Execute() {
	rootCmd.Flags().BoolVarP(&editorFlag, "editor", "e", false, "Open editor to compose prompt")
	rootCmd.Flags().BoolVarP(&interactiveFlag, "interactive", "i", false, "Start interactive chat")
	rootCmd.Flags().BoolVarP(&agentFlag, "agent", "a", false, "Enable agentic capabilities (tools)")
	rootCmd.Flags().BoolVarP(&memoryFlag, "memory", "m", false, "Retain conversation history between turns")
	rootCmd.Flags().IntVar(&stepsFlag, "steps", 10, "Maximum number of agentic steps allowed")
	rootCmd.Flags().Float32VarP(&temperatureFlag, "temperature", "t", 1.0, "Set model temperature (0.0 - 2.0)")
	rootCmd.Flags().StringArrayVar(&mcpFlags, "mcp", []string{}, "Command to start an MCP server (can be used multiple times)")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
