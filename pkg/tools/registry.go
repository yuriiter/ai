package tools

import (
	"encoding/json"
	"fmt"
	"github.com/yuriiter/ai/pkg/mcp"

	openai "github.com/sashabaranov/go-openai"
)

type ToolType int

const (
	TypeInternal ToolType = iota
	TypeMCP
)

type ToolEntry struct {
	Type       ToolType
	Definition openai.FunctionDefinition
	InternalFn func(args string) (string, error)
	MCPClient  *mcp.Client
}

type Registry struct {
	tools []ToolEntry
}

func NewRegistry() *Registry {
	return &Registry{
		tools: make([]ToolEntry, 0),
	}
}

func (r *Registry) LoadMCPTools(command string) error {
	client, err := mcp.NewClient(command)
	if err != nil {
		return err
	}

	resBytes, err := client.Call("tools/list", nil)
	if err != nil {
		client.Close()
		return err
	}

	var result struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"inputSchema"`
		} `json:"tools"`
	}

	if err := json.Unmarshal(resBytes, &result); err != nil {
		client.Close()
		return err
	}

	for _, t := range result.Tools {
		cleanSchema := sanitizeSchema(t.InputSchema)

		r.tools = append(r.tools, ToolEntry{
			Type: TypeMCP,
			Definition: openai.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  cleanSchema,
			},
			MCPClient: client,
		})
	}

	return nil
}

func sanitizeSchema(raw json.RawMessage) json.RawMessage {
	defaultSchema := json.RawMessage(`{"type": "object", "properties": {}, "additionalProperties": false}`)

	if len(raw) == 0 {
		return defaultSchema
	}

	var schemaMap map[string]interface{}
	if err := json.Unmarshal(raw, &schemaMap); err != nil {
		return defaultSchema
	}

	delete(schemaMap, "$schema")
	delete(schemaMap, "title")

	if _, ok := schemaMap["type"]; !ok {
		schemaMap["type"] = "object"
	}

	if _, ok := schemaMap["properties"]; !ok {
		schemaMap["properties"] = map[string]interface{}{}
	}

	cleanBytes, _ := json.Marshal(schemaMap)
	return cleanBytes
}
func (r *Registry) GetOpenAITools() []openai.Tool {
	var apiTools []openai.Tool
	for _, t := range r.tools {
		apiTools = append(apiTools, openai.Tool{
			Type:     openai.ToolTypeFunction,
			Function: &t.Definition,
		})
	}
	return apiTools
}

func (r *Registry) Execute(name string, argsJSON string) (string, error) {
	for _, t := range r.tools {
		if t.Definition.Name == name {
			if t.Type == TypeInternal {
				return t.InternalFn(argsJSON)
			}

			if t.Type == TypeMCP {
				var argsMap map[string]interface{}

				if argsJSON == "" || argsJSON == "null" {
					argsMap = make(map[string]interface{})
				} else {
					if err := json.Unmarshal([]byte(argsJSON), &argsMap); err != nil {
						return "", fmt.Errorf("invalid json args from model: %w", err)
					}
				}

				if argsMap == nil {
					argsMap = make(map[string]interface{})
				}

				callParams := map[string]interface{}{
					"name":      name,
					"arguments": argsMap,
				}

				resBytes, err := t.MCPClient.Call("tools/call", callParams)
				if err != nil {
					return "", err
				}

				var output struct {
					Content []struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"content"`
					IsError bool `json:"isError"`
				}

				if err := json.Unmarshal(resBytes, &output); err != nil {
					return "", fmt.Errorf("failed to parse mcp response: %w", err)
				}

				if output.IsError {
					if len(output.Content) > 0 {
						return fmt.Sprintf("Tool Error: %s", output.Content[0].Text), nil
					}
					return "Tool failed with unspecified error", nil
				}

				if len(output.Content) > 0 {
					return output.Content[0].Text, nil
				}
				return "success", nil
			}
		}
	}
	return "", fmt.Errorf("tool %s not found", name)
}

func (r *Registry) Close() {
	seen := make(map[*mcp.Client]bool)
	for _, t := range r.tools {
		if t.Type == TypeMCP && t.MCPClient != nil && !seen[t.MCPClient] {
			t.MCPClient.Close()
			seen[t.MCPClient] = true
		}
	}
}
