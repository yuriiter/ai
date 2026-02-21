# AI CLI Tool

`ai` is a powerful, extensible command-line interface for interacting with OpenAI-compatible APIs. Beyond simple prompting, it features a full **Agentic Mode** with support for the **Model Context Protocol (MCP)**, **Local RAG (Retrieval-Augmented Generation)**, **Voice Interaction**, and more. 

## Features

*   **Agentic Capabilities:** Enable the AI to perform multi-step tasks using tools (`-a` flag).
*   **MCP Support:** Connect to any [Model Context Protocol](https://modelcontextprotocol.io/) server to give the AI access to local resources (databases, APIs, etc.).
*   **Local RAG:** Chat with your local documents (`.pdf`, `.docx`, `.xlsx`, `.epub`, `.md`, etc.) using local embedding models for privacy and speed (`--rag`).
*   **Voice Mode (not implemented yet, WIP):** Talk to the AI and hear its responses using OpenAI Whisper and TTS (`--voice`).
*   **Context Inclusion:** Easily inject local files as context via glob patterns (`--glob`).
*   **Session Management:** Save and load your chat history to/from Markdown files (`--save-session`, `--session`).
*   **Interactive Chat:** Continuous conversational workflows with memory (`-i`, `-m`).
*   **Flexible Input:** Standard input (stdin) piping, command-line arguments, and external editor integration (`-e`).

## Installation

### Prerequisites

*   Go 1.21+
*   An OpenAI API Key (or compatible provider).
*   **(Optional) MCP:** `npx` or other runtimes if you plan to use specific MCP servers.
*   **(Optional) Voice Mode:** Requires the `portaudio` C library (e.g., `brew install portaudio` on macOS, `sudo apt-get install portaudio19-dev` on Debian/Ubuntu). Linux users may also need an audio player like `mpg123` or `ffmpeg` installed for playback.

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

### Context Inclusion
Easily dump files directly into the AI's context window.

```bash
ai --glob "*.go" "Find any bugs in these files"
```

### RAG (Chat with Documents)
Use `--rag` to index and search through large documents locally. The tool automatically extracts text, generates local embeddings (`sentence-transformers`), and caches them for fast repeated use.

```bash
ai --rag "docs/**/*.md" --rag "*.pdf" -i
```

### Voice Mode
Talk to your agent! Press SPACE to start recording and SPACE again to send. The AI will speak its response back to you.

```bash
ai -im --voice
```

### Session Management
Save your conversation to a Markdown file to resume later or keep a record.

```bash
# Save a session
ai -im --save-session chat.md

# Resume a session
ai -im --session chat.md
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
| `--editor` | `-e` | Open editor to compose prompt. |
| `--glob` | | Glob patterns to include files as full text context. |
| `--interactive` | `-i` | Start interactive chat mode. |
| `--mcp` | | Command to start an MCP server (can be used multiple times). |
| `--memory` | `-m` | Retain conversation history between turns (useful in scripts). |
| `--rag` | | Glob patterns for RAG documents (can be used multiple times). |
| `--rag-top` | | Number of RAG context chunks to retrieve (default: 3). |
| `--save-session` | | Save chat history to a Markdown file. |
| `--session` | | Load chat history from a Markdown file. |
| `--steps` | | Maximum number of agentic steps allowed (default: 10). |
| `--temperature` | `-t` | Set model temperature (0.0 - 2.0). |
| `--voice` | | Enable voice interaction (requires `--interactive`). |

## Development

1. Clone the repository.
2. Ensure you have `portaudio` installed on your system.
3. Build the binary:

```bash
go build -o ai main.go
```
