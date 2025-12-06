# AI CLI Tool

`ai` is a simple, powerful command-line interface tool for interacting with OpenAI-compatible APIs (like OpenAI, Azure, or self-hosted models). It supports streaming, configuration via environment variables, and flexible input handling (arguments, stdin piping, and editor integration).

## Features

*   **API Compatibility:** Works with any API endpoint supported by the `go-openai` library.
*   **Flexible Input:** Get prompts from arguments, piped content, or an external editor.
*   **System Instructions:** Configure the AI's personality or instructions using an environment variable.
*   **Streaming Output:** Get responses in real-time directly to your terminal.

## Installation

### Prerequisites

You need Go (1.18+) installed to build the tool.

### Build and Install

```bash
# Clone the repository (assuming this code is saved in a repo)
# git clone <repo-url>
# cd <repo-dir>

# Build the executable
go build -o ai .

# Move the executable to a directory in your PATH (e.g., /usr/local/bin)
sudo mv ai /usr/local/bin/
```

## Configuration

The tool relies entirely on environment variables for configuration.

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `AI_API_KEY` | **Required.** Your API key (e.g., OpenAI API Key). | None |
| `AI_API_BASE_URL` | Optional. The base URL for the API endpoint (useful for services other than OpenAI). | `https://api.openai.com/v1` |
| `AI_API_MODEL` | Optional. The specific model to use for chat completion. | `gpt-3.5-turbo` |
| `AI_SYSTEM_INSTRUCTIONS` | Optional. A system prompt to guide the AI's behavior and personality for every request. | None |
| `EDITOR` | Optional. The command to launch your preferred text editor (used with the `-e` flag). | `vim`, `nano`, or `vi` |

### Example Setup

You should set these variables in your shell configuration (`.bashrc`, `.zshrc`, etc.) or export them for the current session.

```bash
export AI_API_KEY="sk-..."
export AI_API_MODEL="gpt-4o"
export AI_SYSTEM_INSTRUCTIONS="You are a helpful assistant who always answers in concise, markdown-formatted bullet points."
```

## Usage

The primary usage is simply providing a prompt as command arguments.

```bash
ai Explain the concept of monads in functional programming
```

### Input Methods

#### 1. Arguments

The simplest way is passing the prompt directly:

```bash
ai Write a Python function to reverse a string
```

#### 2. Stdin Piping

You can pipe content from other commands into `ai`. This is ideal for analyzing files or code snippets.

```bash
cat my_script.py | ai Review this Python script for security vulnerabilities
```

You can combine a prompt argument with piped data:

```bash
ls -l | ai Analyze the output below and summarize the file permissions
```

#### 3. Editor Mode (`-e`)

Use the `-e` or `--editor` flag to launch your configured external editor (defined by `$EDITOR` or the default). This is useful for writing multi-line or complex prompts.

```bash
ai -e
```

If you provide arguments or pipe data *along with* the `-e` flag, that content will be pre-pended to the editor's output, allowing you to give instructions and elaborate on the piped content inside the editor.

### Example: Using System Instructions

If you have configured `AI_SYSTEM_INSTRUCTIONS` (e.g., set to make the AI a pirate), the command will automatically follow those instructions:

```bash
# Assuming AI_SYSTEM_INSTRUCTIONS="You are a friendly pirate. Format your message with ANSI"
ai Tell me about the weather

# Colorful terminal output: "Arrr, the skies be clear, captain! Perfect weather for sailing the high seas!"
```

## Development

The tool is written in Go. If you modify the source code, you can rebuild it using:

```bash
go build -o ai
```
