package cmd

import (
	"bufio"
	"context"
	"fmt"
	"github.com/yuriiter/ai/pkg/agent"
	"github.com/yuriiter/ai/pkg/config"
	"github.com/yuriiter/ai/pkg/ui"
	"github.com/yuriiter/ai/pkg/voice"
	"golang.org/x/term"
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
	ragFlags        []string
	ragTopKFlag     int
	saveSessionFlag string
	loadSessionFlag string
	voiceFlag       bool
)

var rootCmd = &cobra.Command{
	Use:   "ai [prompt...]",
	Short: "A CLI AI Agent with optional MCP and RAG support",
	Run: func(cmd *cobra.Command, args []string) {
		cfg := config.Load()

		cfg.MaxSteps = stepsFlag
		cfg.RetainHistory = memoryFlag
		cfg.Temperature = temperatureFlag
		cfg.RagGlobs = ragFlags
		cfg.RagTopK = ragTopKFlag

		aiAgent, err := agent.New(cfg, agentFlag, mcpFlags)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sError initializing agent: %v%s\n", ui.ColorRed, err, ui.ColorReset)
			os.Exit(1)
		}
		defer aiAgent.Close()

		if loadSessionFlag != "" {
			if err := aiAgent.LoadSession(loadSessionFlag); err != nil {
				fmt.Fprintf(os.Stderr, "%sError loading session: %v%s\n", ui.ColorRed, err, ui.ColorReset)
				os.Exit(1)
			}
			fmt.Printf("%sSession loaded from %s%s\n", ui.ColorGreen, loadSessionFlag, ui.ColorReset)
		}

		if saveSessionFlag != "" {
			defer func() {
				if err := aiAgent.SaveSession(saveSessionFlag); err != nil {
					fmt.Fprintf(os.Stderr, "%sError saving session: %v%s\n", ui.ColorRed, err, ui.ColorReset)
				} else {
					fmt.Printf("%sSession saved to %s%s\n", ui.ColorGreen, saveSessionFlag, ui.ColorReset)
				}
			}()
		}

		ctx := context.Background()

		if len(ragFlags) > 0 {
			if err := aiAgent.InitializeRAG(ctx); err != nil {
				fmt.Fprintf(os.Stderr, "%sRAG Initialization Error: %v%s\n", ui.ColorRed, err, ui.ColorReset)
				os.Exit(1)
			}
		}

		if interactiveFlag {
			if voiceFlag {
				startVoiceInteractive(ctx, aiAgent)
			} else {
				startInteractive(ctx, aiAgent)
			}
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

func startVoiceInteractive(ctx context.Context, ai *agent.Agent) {
	fmt.Println("Voice Mode Enabled.")
	fmt.Println("Press SPACE to start recording. Press SPACE again to stop and send.")
	fmt.Println("Press Ctrl+C to quit.")

	vm, err := voice.NewManager(config.Load().ApiKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to init voice manager: %v\n", err)
		os.Exit(1)
	}
	defer vm.Close()

	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to set raw terminal: %v\n", err)
		os.Exit(1)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	screenReader := bufio.NewReader(os.Stdin)

	for {
		fmt.Printf("\r\033[K[WAITING] Press SPACE to speak...")

		for {
			r, _, err := screenReader.ReadRune()
			if err != nil {
				return
			}
			if r == ' ' {
				break
			}
			if r == 3 {
				return
			}
		}

		fmt.Printf("\r\033[K[RECORDING] Speak now (Press SPACE to stop)...")

		audioData, err := vm.RecordUntilSpace(screenReader)
		if err != nil {
			fmt.Printf("\r\033[KError recording: %v\n", err)
			continue
		}

		fmt.Printf("\r\033[K[PROCESSING] Transcribing...")
		text, err := vm.Transcribe(ctx, audioData)
		if err != nil {
			fmt.Printf("\r\033[KTranscription error: %v\n", err)
			continue
		}

		if strings.TrimSpace(text) == "" {
			fmt.Printf("\r\033[KNo speech detected.\n")
			continue
		}

		term.Restore(int(os.Stdin.Fd()), oldState)
		fmt.Printf("\r\033[K\n%sYou (Voice): %s%s\n", ui.ColorBlue, text, ui.ColorReset)

		response, err := ai.RunTurnCapture(ctx, text)
		term.MakeRaw(int(os.Stdin.Fd()))

		if err != nil {
			fmt.Printf("Agent Error: %v\n", err)
			continue
		}

		fmt.Printf("\r\033[K[SPEAKING] Generating audio...")
		if err := vm.Speak(ctx, response); err != nil {
			fmt.Printf("\r\033[KError speaking: %v\n", err)
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
	rootCmd.Flags().StringArrayVar(&mcpFlags, "mcp", []string{}, "Command to start an MCP server")
	rootCmd.Flags().StringArrayVar(&ragFlags, "rag", []string{}, "Glob patterns for RAG documents (can be used multiple times)")
	rootCmd.Flags().IntVar(&ragTopKFlag, "rag-top", 3, "Number of RAG context chunks to retrieve")
	rootCmd.Flags().StringVar(&saveSessionFlag, "save-session", "", "Save chat history to a Markdown file")
	rootCmd.Flags().StringVar(&loadSessionFlag, "session", "", "Load chat history from a Markdown file")
	rootCmd.Flags().BoolVar(&voiceFlag, "voice", false, "Enable voice interaction (requires --interactive)")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
