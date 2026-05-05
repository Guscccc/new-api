package service

import (
	"testing"

	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/stretchr/testify/require"
)

func newClaudeConvertTestInfo() *relaycommon.RelayInfo {
	return &relaycommon.RelayInfo{
		SendResponseCount: 1,
		ClaudeConvertInfo: &relaycommon.ClaudeConvertInfo{
			LastMessagesType: relaycommon.LastMessageTypeNone,
		},
	}
}

func TestStreamResponseOpenAI2ClaudeConvertsThinkTagsToThinkingBlocks(t *testing.T) {
	info := newClaudeConvertTestInfo()
	content := "<think>inspect file</think>visible text"
	finishReason := "stop"
	streamResponse := &dto.ChatCompletionsStreamResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.ChatCompletionsStreamResponseChoice{{
			Delta: dto.ChatCompletionsStreamResponseChoiceDelta{
				Content: &content,
			},
			FinishReason: &finishReason,
		}},
		Usage: &dto.Usage{},
	}

	claudeResponses := StreamResponseOpenAI2Claude(streamResponse, info)

	var thinkingDelta string
	var textDelta string
	for _, response := range claudeResponses {
		if response.Delta == nil {
			continue
		}
		switch response.Delta.Type {
		case "thinking_delta":
			thinkingDelta += *response.Delta.Thinking
		case "text_delta":
			textDelta += *response.Delta.Text
		}
	}
	require.Equal(t, "inspect file", thinkingDelta)
	require.Equal(t, "visible text", textDelta)
	require.NotContains(t, textDelta, "<think>")
}

func TestStreamResponseOpenAI2ClaudePreservesMidMessageThinkTagsAsText(t *testing.T) {
	info := newClaudeConvertTestInfo()
	content := "before <think>not reasoning metadata</think> after"
	finishReason := "stop"
	streamResponse := &dto.ChatCompletionsStreamResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.ChatCompletionsStreamResponseChoice{{
			Delta: dto.ChatCompletionsStreamResponseChoiceDelta{
				Content: &content,
			},
			FinishReason: &finishReason,
		}},
		Usage: &dto.Usage{},
	}

	claudeResponses := StreamResponseOpenAI2Claude(streamResponse, info)

	var thinkingDelta string
	var textDelta string
	for _, response := range claudeResponses {
		if response.Delta == nil {
			continue
		}
		switch response.Delta.Type {
		case "thinking_delta":
			thinkingDelta += *response.Delta.Thinking
		case "text_delta":
			textDelta += *response.Delta.Text
		}
	}
	require.Empty(t, thinkingDelta)
	require.Equal(t, content, textDelta)
}

func TestResponseOpenAI2ClaudePreservesMidMessageThinkTagsAsText(t *testing.T) {
	content := "before <think>reasoning</think> after"
	openAIResponse := &dto.OpenAITextResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.OpenAITextResponseChoice{{
			Message: dto.Message{Role: "assistant"},
		}},
	}
	openAIResponse.Choices[0].Message.SetStringContent(content)

	claudeResponse := ResponseOpenAI2Claude(openAIResponse, &relaycommon.RelayInfo{})

	require.Len(t, claudeResponse.Content, 1)
	require.Equal(t, "text", claudeResponse.Content[0].Type)
	require.Equal(t, content, claudeResponse.Content[0].GetText())
}

func TestResponseOpenAI2ClaudeConvertsLeadingThinkTagsToThinkingBlocks(t *testing.T) {
	content := "<think>reasoning</think> after"
	openAIResponse := &dto.OpenAITextResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.OpenAITextResponseChoice{{
			Message: dto.Message{Role: "assistant"},
		}},
	}
	openAIResponse.Choices[0].Message.SetStringContent(content)

	claudeResponse := ResponseOpenAI2Claude(openAIResponse, &relaycommon.RelayInfo{})

	require.Len(t, claudeResponse.Content, 2)
	require.Equal(t, "thinking", claudeResponse.Content[0].Type)
	require.Equal(t, "reasoning", *claudeResponse.Content[0].Thinking)
	require.Equal(t, "text", claudeResponse.Content[1].Type)
	require.Equal(t, " after", claudeResponse.Content[1].GetText())
}

func TestClaudeToOpenAIRequestMapsCacheControlToPromptCacheRetention(t *testing.T) {
	request := dto.ClaudeRequest{
		Model:        "claude-3-5-sonnet",
		CacheControl: []byte(`{"type":"ephemeral"}`),
		Messages: []dto.ClaudeMessage{{
			Role:    "user",
			Content: "hello",
		}},
	}

	openAIRequest, err := ClaudeToOpenAIRequest(request, &relaycommon.RelayInfo{})

	require.NoError(t, err)
	require.Equal(t, `"in_memory"`, string(openAIRequest.PromptCacheRetention))
	require.Empty(t, openAIRequest.Messages[0].ParseContent()[0].CacheControl)
}

func TestClaudeToOpenAIRequestMapsOneHourCacheControlToPromptCacheRetention(t *testing.T) {
	request := dto.ClaudeRequest{
		Model:        "claude-3-5-sonnet",
		CacheControl: []byte(`{"type":"ephemeral","ttl":"1h"}`),
		Messages: []dto.ClaudeMessage{{
			Role:    "user",
			Content: "hello",
		}},
	}

	openAIRequest, err := ClaudeToOpenAIRequest(request, &relaycommon.RelayInfo{})

	require.NoError(t, err)
	require.Equal(t, `"24h"`, string(openAIRequest.PromptCacheRetention))
}

func TestClaudeToOpenAIRequestKeepsLongestCacheRetention(t *testing.T) {
	request := dto.ClaudeRequest{
		Model:        "claude-3-5-sonnet",
		CacheControl: []byte(`{"type":"ephemeral"}`),
		System: []any{map[string]any{
			"type":          "text",
			"text":          "system",
			"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
		}},
		Messages: []dto.ClaudeMessage{{
			Role:    "user",
			Content: "hello",
		}},
	}

	openAIRequest, err := ClaudeToOpenAIRequest(request, &relaycommon.RelayInfo{})

	require.NoError(t, err)
	require.Equal(t, `"24h"`, string(openAIRequest.PromptCacheRetention))
}

func TestClaudeToOpenAIRequestHandlesNilRelayInfoWithStructuredSystem(t *testing.T) {
	request := dto.ClaudeRequest{
		Model: "claude-3-5-sonnet",
		System: []any{map[string]any{
			"type": "text",
			"text": "system",
		}},
		Messages: []dto.ClaudeMessage{{
			Role:    "user",
			Content: "hello",
		}},
	}

	openAIRequest, err := ClaudeToOpenAIRequest(request, nil)

	require.NoError(t, err)
	require.Len(t, openAIRequest.Messages, 2)
}

func TestStreamResponseOpenAI2ClaudeSanitizesEmptyOptionalToolArgs(t *testing.T) {
	info := newClaudeConvertTestInfo()
	info.Request = &dto.ClaudeRequest{
		Tools: []dto.Tool{{
			Name: "Read",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{"type": "string"},
					"pages":     map[string]any{"type": "string"},
				},
				"required": []any{"file_path"},
			},
		}},
	}
	finishReason := "tool_calls"
	streamResponse := &dto.ChatCompletionsStreamResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.ChatCompletionsStreamResponseChoice{{
			Delta: dto.ChatCompletionsStreamResponseChoiceDelta{
				ToolCalls: []dto.ToolCallResponse{{
					ID: "call_1",
					Function: dto.FunctionResponse{
						Name:      "Read",
						Arguments: `{"file_path":"notes.md","pages":""}`,
					},
				}},
			},
			FinishReason: &finishReason,
		}},
		Usage: &dto.Usage{},
	}

	claudeResponses := StreamResponseOpenAI2Claude(streamResponse, info)

	var partialJSON string
	for _, response := range claudeResponses {
		if response.Delta != nil && response.Delta.Type == "input_json_delta" {
			partialJSON = *response.Delta.PartialJson
		}
	}
	require.JSONEq(t, `{"file_path":"notes.md"}`, partialJSON)
}

func TestStreamResponseOpenAI2ClaudeHandlesCumulativeToolArgumentSnapshots(t *testing.T) {
	info := newClaudeConvertTestInfo()
	finishReason := "tool_calls"
	toolIndex := 0
	chunks := []string{
		`{"file_path":"notes.md","offset":1`,
		`{"file_path":"notes.md","offset":10`,
		`{"file_path":"notes.md","offset":100,"limit":5}`,
	}

	var claudeResponses []*dto.ClaudeResponse
	for i, arguments := range chunks {
		if i > 0 {
			info.SendResponseCount++
		}
		toolCall := dto.ToolCallResponse{
			Index: &toolIndex,
			ID:    "call_1",
			Function: dto.FunctionResponse{
				Arguments: arguments,
			},
		}
		if i == 0 {
			toolCall.Function.Name = "Read"
		}
		var chunkFinishReason *string
		var usage *dto.Usage
		if i == len(chunks)-1 {
			chunkFinishReason = &finishReason
			usage = &dto.Usage{}
		}
		streamResponse := &dto.ChatCompletionsStreamResponse{
			Id:    "chatcmpl-1",
			Model: "claude-3-5-sonnet",
			Choices: []dto.ChatCompletionsStreamResponseChoice{{
				Delta: dto.ChatCompletionsStreamResponseChoiceDelta{
					ToolCalls: []dto.ToolCallResponse{toolCall},
				},
				FinishReason: chunkFinishReason,
			}},
			Usage: usage,
		}
		claudeResponses = append(claudeResponses, StreamResponseOpenAI2Claude(streamResponse, info)...)
	}

	var partialJSON string
	for _, response := range claudeResponses {
		if response.Delta != nil && response.Delta.Type == "input_json_delta" {
			partialJSON += *response.Delta.PartialJson
		}
	}
	require.JSONEq(t, `{"file_path":"notes.md","offset":100,"limit":5}`, partialJSON)
}

func TestMergeToolCallArgumentBufferHandlesFullSnapshotReplacement(t *testing.T) {
	current := `{"file_path":"notes.md","offset":1000,"limit":5}`
	incoming := `{"file_path":"notes.md","offset":100,"limit":5}`

	merged := mergeToolCallArgumentBuffer(current, incoming)

	require.JSONEq(t, incoming, merged)
}

func TestMergeToolCallArgumentBufferIgnoresOlderShorterSnapshot(t *testing.T) {
	current := `{"file_path":"notes.md","offset":100,"limit":5}`
	incoming := `{"file_path":"notes.md","offset":100}`

	merged := mergeToolCallArgumentBuffer(current, incoming)

	require.JSONEq(t, current, merged)
}

func TestMergeToolCallArgumentBufferAppendsTrueDelta(t *testing.T) {
	current := `{"file_path":"notes.md","offset":10`
	incoming := `0,"limit":5}`

	merged := mergeToolCallArgumentBuffer(current, incoming)

	require.JSONEq(t, `{"file_path":"notes.md","offset":100,"limit":5}`, merged)
}

func TestMergeToolCallArgumentBufferAppendsNestedObjectDelta(t *testing.T) {
	current := `{"outer":`
	incoming := `{"inner":1}}`

	merged := mergeToolCallArgumentBuffer(current, incoming)

	require.JSONEq(t, `{"outer":{"inner":1}}`, merged)
}

func TestResponseOpenAI2ClaudeSanitizesEmptyOptionalToolArgs(t *testing.T) {
	info := &relaycommon.RelayInfo{}
	openAIResponse := &dto.OpenAITextResponse{
		Id:    "chatcmpl-1",
		Model: "claude-3-5-sonnet",
		Choices: []dto.OpenAITextResponseChoice{{
			FinishReason: "tool_calls",
			Message:      dto.Message{},
		}},
	}
	openAIResponse.Choices[0].Message.SetToolCalls([]dto.ToolCallRequest{{
		ID:   "call_1",
		Type: "function",
		Function: dto.FunctionRequest{
			Name:      "Read",
			Arguments: `{"file_path":"notes.md","pages":""}`,
		},
	}})
	info.Request = &dto.ClaudeRequest{
		Tools: []dto.Tool{{
			Name: "Read",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{"type": "string"},
					"pages":     map[string]any{"type": "string"},
				},
				"required": []any{"file_path"},
			},
		}},
	}

	claudeResponse := ResponseOpenAI2Claude(openAIResponse, info)

	require.Len(t, claudeResponse.Content, 1)
	require.Equal(t, "tool_use", claudeResponse.Content[0].Type)
	require.Equal(t, map[string]any{"file_path": "notes.md"}, claudeResponse.Content[0].Input)
}

func TestClaudeToOpenAIRequestKeepsCacheControlForOpenRouterClaude(t *testing.T) {
	request := dto.ClaudeRequest{
		Model:        "claude-3-5-sonnet",
		CacheControl: []byte(`{"type":"ephemeral"}`),
		Messages: []dto.ClaudeMessage{{
			Role: "user",
			Content: []any{map[string]any{
				"type":          "text",
				"text":          "hello",
				"cache_control": map[string]any{"type": "ephemeral"},
			}},
		}},
	}
	info := &relaycommon.RelayInfo{ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenRouter, UpstreamModelName: "anthropic/claude-3-5-sonnet"}}

	openAIRequest, err := ClaudeToOpenAIRequest(request, info)

	require.NoError(t, err)
	require.Empty(t, openAIRequest.PromptCacheRetention)
	require.Len(t, openAIRequest.Messages, 1)
	require.Len(t, openAIRequest.Messages[0].ParseContent(), 1)
	require.NotEmpty(t, openAIRequest.Messages[0].ParseContent()[0].CacheControl)
}
