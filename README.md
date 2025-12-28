# AI CLI Tool

`ai` is a powerful, extensible command-line interface for interacting with OpenAI-compatible APIs. Beyond simple prompting, it features a full **Agentic Mode** with support for the **Model Context Protocol (MCP)**, allowing the AI to use tools, interact with your filesystem, and connect to external servers.

## Features

*   **Agentic Capabilities:** Enable the AI to perform multi-step tasks using tools (`-a` flag).
*   **MCP Support:** Connect to any [Model Context Protocol](https://modelcontextprotocol.io/) server to give the AI access to local resources (databases, filesystems, etc.).
*   **Interactive Chat:** specific interactive mode (`-i`) for conversational workflows.
*   **Flexible Input:** Standard input (stdin) piping, command-line arguments, and external editor integration (`-e`).
*   **Customizable:** Configure models, temperature, and system instructions via environment variables or flags.

## Installation

### Prerequisites

*   Go 1.21+
*   An OpenAI API Key (or compatible provider).
*   (Optional) `npx` or other runtimes if you plan to use specific MCP servers.

### Install

```bash
go install github.com/yuriiter/ai@latest
```

## Configuration

The tool uses environment variables for default configuration.

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Required.** Your API key. | None |
| `OPENAI_BASE_URL` | Optional. Base URL for the API (useful for Ollama, Azure, etc.). | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Optional. The specific model to use. | `gpt-4o` |
| `OPENAI_SYSTEM_INSTRUCTIONS` | Optional. Default system prompt/persona. | Built-in helper persona |
| `OPENAI_TEMPERATURE` | Optional. Default temperature (creativity). | `1.0` |
| `EDITOR` | Optional. Editor for the `-e` flag. | `vim`, `nano`, or `vi` |

## Usage

### Basic Prompting
Just like `echo`, you can pass arguments directly:

```bash
ai Explain the concept of recursion
```

### Interactive Mode
Start a chat session with memory:

```bash
ai -im
```

### Agentic Mode & MCP (Model Context Protocol)
The real power of `ai` comes from connecting it to MCP servers. This allows the AI to "do" things rather than just talk.

To use tools, you must enable agent mode (`-a`) and provide an MCP server command (`--mcp`).

**Example: Giving the AI access to your filesystem**
(Requires `npx` installed)

```bash
ai -a --mcp "npx -y @modelcontextprotocol/server-filesystem ." \
   "Read the file 'main.go' and tell me what the package name is"
```

You can chain multiple MCP servers:

```bash
ai -a \
   --mcp "npx -y @modelcontextprotocol/server-filesystem ." \
   --mcp "python3 my_custom_server.py" \
   "Analyze my files and upload the summary to my custom server"
```

### Using the Editor
Use `-e` to open your default text editor (Vim/Nano) to compose complex prompts. If you pipe data in, it will appear in the editor for you to annotate.

```bash
git diff | ai -e
# Opens editor with the diff, letting you add: "Write a commit message for these changes"
```

### Flags Reference

| Flag | Short | Description |
| :--- | :--- | :--- |
| `--agent` | `-a` | Enable agentic capabilities (required for MCP tools). |
| `--interactive` | `-i` | Start interactive chat mode. |
| `--mcp` | | Command to start an MCP server (can be used multiple times). |
| `--editor` | `-e` | Open editor to compose prompt. |
| `--memory` | `-m` | Retain conversation history between turns (useful in scripts). |
| `--steps` | | Maximum number of agentic steps allowed (default: 10). |
| `--temperature` | `-t` | Set model temperature (0.0 - 2.0). |

## Development

1. Clone the repository.
2. Build the binary:

```bash
go build -o ai main.go
```
