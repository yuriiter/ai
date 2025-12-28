package agent

import (
	"context"
	"errors"
	"fmt"
	"github.com/yuriiter/ai/pkg/config"
	"github.com/yuriiter/ai/pkg/tools"
	"github.com/yuriiter/ai/pkg/ui"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

type Agent struct {
	client      *openai.Client
	config      config.Config
	history     []openai.ChatCompletionMessage
	Registry    *tools.Registry
	agenticMode bool
}

func New(cfg config.Config, agenticMode bool, mcpServers []string) (*Agent, error) {
	clientConfig := openai.DefaultConfig(cfg.ApiKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}

	reg := tools.NewRegistry()

	if agenticMode {
		for _, serverCmd := range mcpServers {
			if serverCmd == "" {
				continue
			}
			fmt.Printf("%sConnecting to MCP: %s...%s\n", ui.ColorBlue, serverCmd, ui.ColorReset)
			if err := reg.LoadMCPTools(serverCmd); err != nil {
				return nil, fmt.Errorf("failed to load MCP server '%s': %w", serverCmd, err)
			}
		}

		toolsList := reg.GetOpenAITools()
		var names []string
		for _, t := range toolsList {
			names = append(names, t.Function.Name)
		}
		if len(names) > 0 {
			fmt.Printf("%sLoaded Tools: %s%s\n", ui.ColorGreen, strings.Join(names, ", "), ui.ColorReset)
		}
	}

	sysPrompt := cfg.SystemInstructions
	if sysPrompt == "" {
		if agenticMode {
			sysPrompt = "You are a helpful assistant with access to tools.\n" +
				"IMPORTANT GUIDELINES FOR TOOL USE:\n" +
				"1. Use tools only when needed. For general conversation or greetings, do not use tools.\n" +
				"2. FORMATTING IS CRITICAL: When calling a tool, use ONLY the tool name (e.g., 'get_weather').\n" +
				"   NEVER append JSON arguments to the tool name. (e.g., DO NOT write 'get_weather{\"city\": \"London\"}').\n" +
				"   Put all arguments inside the JSON arguments object.\n" +
				"3. Do not guess argument values. If a tool requires a specific ID, path, or parameter you do not have:\n" +
				"   a. Check if another tool can provide this information.\n" +
				"   b. If not, ask the user for clarification.\n" +
				"4. Always provide all required parameters defined in the tool schema."
		} else {
			sysPrompt = "You are a helpful assistant."
		}
	}

	agent := &Agent{
		client:      openai.NewClientWithConfig(clientConfig),
		config:      cfg,
		history:     make([]openai.ChatCompletionMessage, 0),
		Registry:    reg,
		agenticMode: agenticMode,
	}

	if sysPrompt != "" {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: sysPrompt,
		})
	}

	return agent, nil
}

func (a *Agent) Close() {
	if a.Registry != nil {
		a.Registry.Close()
	}
}

func (a *Agent) pruneHistory() {
	const maxHistory = 10
	if len(a.history) <= maxHistory {
		return
	}

	var newHistory []openai.ChatCompletionMessage
	if len(a.history) > 0 && a.history[0].Role == openai.ChatMessageRoleSystem {
		newHistory = append(newHistory, a.history[0])
		remaining := a.history[len(a.history)-(maxHistory-1):]
		newHistory = append(newHistory, remaining...)
	} else {
		newHistory = a.history[len(a.history)-maxHistory:]
	}
	a.history = newHistory
}

func (a *Agent) RunTurn(ctx context.Context, prompt string, streaming bool) error {
	historyStartLen := len(a.history)

	defer func() {
		if !a.config.RetainHistory {
			a.history = a.history[:historyStartLen]
		}
	}()

	a.pruneHistory()

	a.history = append(a.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	maxSteps := a.config.MaxSteps
	if !a.agenticMode {
		maxSteps = 1
	}

	steps := 0
	for steps < maxSteps {
		req := openai.ChatCompletionRequest{
			Model:       a.config.Model,
			Messages:    a.history,
			Temperature: a.config.Temperature,
		}

		if a.agenticMode {
			availTools := a.Registry.GetOpenAITools()
			if len(availTools) > 0 {
				req.Tools = availTools
			}
		}

		resp, err := a.client.CreateChatCompletion(ctx, req)
		if err != nil {
			return fmt.Errorf("api error: %w", err)
		}

		msg := resp.Choices[0].Message
		a.history = append(a.history, msg)

		if len(msg.ToolCalls) > 0 && a.agenticMode {
			ui.PrintToolUse(msg.ToolCalls[0].Function.Name, msg.ToolCalls[0].Function.Arguments)

			for _, toolCall := range msg.ToolCalls {
				cleanName := strings.Split(toolCall.Function.Name, "{")[0]
				cleanName = strings.Split(cleanName, "=")[0]
				cleanName = strings.TrimSpace(cleanName)

				output, err := a.Registry.Execute(cleanName, toolCall.Function.Arguments)
				if err != nil {
					output = fmt.Sprintf("Error executing tool: %v", err)
				}

				if len(output) > 10000 {
					output = output[:10000] + "\n...(truncated output)"
				}

				a.history = append(a.history, openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    output,
					ToolCallID: toolCall.ID,
				})
			}
			steps++
			continue
		}

		ui.PrintAgentMessage(msg.Content + "\n")
		return nil
	}

	return errors.New("agent step limit reached")
}
