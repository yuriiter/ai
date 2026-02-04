package config

import (
	"os"
	"os/exec"
	"strconv"
)

type Config struct {
	ApiKey             string
	BaseURL            string
	Model              string
	EmbeddingProvider  string
	EmbeddingModel     string
	Editor             string
	SystemInstructions string
	MaxSteps           int
	RetainHistory      bool
	Temperature        float32
	RagGlob            string
}

func Load() Config {
	c := Config{
		ApiKey:             os.Getenv("OPENAI_API_KEY"),
		BaseURL:            os.Getenv("OPENAI_BASE_URL"),
		Model:              os.Getenv("OPENAI_MODEL"),
		EmbeddingProvider:  os.Getenv("AI_EMBEDDING_PROVIDER"),
		EmbeddingModel:     os.Getenv("OPENAI_EMBEDDING_MODEL"),
		Editor:             os.Getenv("EDITOR"),
		SystemInstructions: os.Getenv("OPENAI_SYSTEM_INSTRUCTIONS"),
		MaxSteps:           10,
		Temperature:        1.0,
	}

	if c.EmbeddingProvider == "" {
		c.EmbeddingProvider = "openai"
	}

	// Default model for OpenAI
	if c.EmbeddingModel == "" {
		c.EmbeddingModel = "text-embedding-3-small"
	}

	if c.Model == "" {
		c.Model = "gpt-4o"
	}

	if val := os.Getenv("OPENAI_TEMPERATURE"); val != "" {
		if f, err := strconv.ParseFloat(val, 32); err == nil {
			c.Temperature = float32(f)
		}
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
