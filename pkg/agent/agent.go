package agent

import (
	"context"
	"errors"
	"fmt"
	"github.com/yuriiter/ai/pkg/config"
	"github.com/yuriiter/ai/pkg/rag"
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
	RagEngine   *rag.Engine
	agenticMode bool
}

func New(cfg config.Config, agenticMode bool, mcpServers []string) (*Agent, error) {
	clientConfig := openai.DefaultConfig(cfg.ApiKey)
	if cfg.BaseURL != "" {
		clientConfig.BaseURL = cfg.BaseURL
	}

	client := openai.NewClientWithConfig(clientConfig)
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
				"   NEVER append JSON arguments to the tool name.\n" +
				"   Put all arguments inside the JSON arguments object.\n" +
				"3. Do not guess argument values.\n" +
				"4. Always provide all required parameters defined in the tool schema."
		} else {
			sysPrompt = "You are a helpful assistant."
		}
	}

	ragEngine, err := rag.New(client, cfg.EmbeddingProvider, cfg.EmbeddingModel)
	if err != nil {
		return nil, fmt.Errorf("failed to init RAG engine: %w", err)
	}

	agent := &Agent{
		client:      client,
		config:      cfg,
		history:     make([]openai.ChatCompletionMessage, 0),
		Registry:    reg,
		agenticMode: agenticMode,
		RagEngine:   ragEngine,
	}

	if sysPrompt != "" {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: sysPrompt,
		})
	}

	return agent, nil
}

func (a *Agent) InitializeRAG(ctx context.Context) error {
	if a.config.RagGlob == "" {
		return nil
	}

	cachePath := rag.GetDefaultCachePath(a.config.RagGlob)

	if a.RagEngine.CacheExists(cachePath) {
		fmt.Printf("%sFound embedding cache, validating...%s\n", ui.ColorBlue, ui.ColorReset)

		valid, reason := a.RagEngine.ValidateCache(cachePath, a.config.RagGlob)

		if valid {
			fmt.Printf("%sCache is valid, loading...%s\n", ui.ColorGreen, ui.ColorReset)
			if _, err := a.RagEngine.LoadEmbeddings(cachePath); err != nil {
				fmt.Printf("%sCache load failed: %v, regenerating...%s\n", ui.ColorRed, err, ui.ColorReset)
			} else {
				return nil
			}
		} else {
			fmt.Printf("%sCache is stale: %s%s\n", ui.ColorRed, reason, ui.ColorReset)
			fmt.Printf("%sRegenerating embeddings...%s\n", ui.ColorBlue, ui.ColorReset)
		}
	} else {
		fmt.Printf("%sNo cache found, generating embeddings...%s\n", ui.ColorBlue, ui.ColorReset)
	}

	if err := a.RagEngine.IngestGlob(ctx, a.config.RagGlob); err != nil {
		return err
	}

	if err := a.RagEngine.SaveEmbeddings(cachePath, a.config.RagGlob, a.config.EmbeddingProvider, a.config.EmbeddingModel); err != nil {
		fmt.Printf("%sWarning: Failed to save cache: %v%s\n", ui.ColorRed, err, ui.ColorReset)
	}

	return nil
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

func (a *Agent) generateSearchKeywords(ctx context.Context, userQuery string) string {
	fmt.Printf("%sGenerating search keywords...%s ", ui.ColorBlue, ui.ColorReset)

	req := openai.ChatCompletionRequest{
		Model: a.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are a search assistant. Convert the user's question into a list of specific search keywords to search the vector database in details. Output ONLY the space-separated keywords, do your best in search assistance, output most relevant keywords and pretty many of them. No explanation.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userQuery,
			},
		},
		Temperature: 0.1,
	}

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil || len(resp.Choices) == 0 {
		fmt.Println("(failed, using original query)")
		return userQuery
	}

	keywords := strings.TrimSpace(resp.Choices[0].Message.Content)
	fmt.Printf("[%s]\n", keywords)
	return keywords
}

func (a *Agent) RunTurn(ctx context.Context, prompt string, streaming bool) error {
	historyStartLen := len(a.history)

	defer func() {
		if !a.config.RetainHistory {
			a.history = a.history[:historyStartLen]
		}
	}()

	a.pruneHistory()

	finalPrompt := prompt

	if a.config.RagGlob != "" && len(a.RagEngine.Chunks) > 0 {
		searchQuery := a.generateSearchKeywords(ctx, prompt)

		results, err := a.RagEngine.Search(ctx, searchQuery, 3)
		if err != nil {
			fmt.Printf("%sRAG Search Error: %v%s\n", ui.ColorRed, err, ui.ColorReset)
		} else if len(results) > 0 {
			var contextBuilder strings.Builder
			contextBuilder.WriteString("Use the following context to answer the user's question:\n\n")
			for _, r := range results {
				contextBuilder.WriteString(fmt.Sprintf("--- Source: %s ---\n%s\n\n", r.Filename, r.Text))
			}
			contextBuilder.WriteString("User Question: " + prompt)
			finalPrompt = contextBuilder.String()
			fmt.Printf("%sFound %d relevant context chunks.%s\n", ui.ColorGreen, len(results), ui.ColorReset)
		}
	}

	a.history = append(a.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: finalPrompt,
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

		if len(resp.Choices) == 0 {
			return fmt.Errorf("api returned empty response (no choices)")
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
