package openaicompat

import (
	"testing"

	"github.com/QuantumNous/new-api/dto"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionsRequestToResponsesRequestPreservesPromptCacheFields(t *testing.T) {
	request := &dto.GeneralOpenAIRequest{
		Model:                "gpt-5",
		PromptCacheKey:       "session-1",
		PromptCacheRetention: []byte(`"in_memory"`),
	}
	message := dto.Message{Role: "user"}
	message.SetStringContent("hello")
	request.Messages = []dto.Message{message}

	responsesRequest, err := ChatCompletionsRequestToResponsesRequest(request)

	require.NoError(t, err)
	require.JSONEq(t, `"session-1"`, string(responsesRequest.PromptCacheKey))
	require.JSONEq(t, `"in_memory"`, string(responsesRequest.PromptCacheRetention))
}

func TestSanitizeResponsesRequestCacheControlRemovesNestedContentFields(t *testing.T) {
	request := &dto.OpenAIResponsesRequest{
		Model: "gpt-5",
		Input: []byte(`[{"role":"user","content":[{"type":"input_text","text":"stable prefix","cache_control":{"type":"ephemeral"}},{"type":"input_text","text":"longer prefix","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]`),
	}

	err := SanitizeResponsesRequestCacheControl(request)

	require.NoError(t, err)
	require.NotContains(t, string(request.Input), "cache_control")
	require.JSONEq(t, `"24h"`, string(request.PromptCacheRetention))
	require.JSONEq(t, `[{"role":"user","content":[{"type":"input_text","text":"stable prefix"},{"type":"input_text","text":"longer prefix"}]}]`, string(request.Input))
}

func TestChatCompletionsRequestToResponsesRequestRemovesNestedCacheControl(t *testing.T) {
	request := &dto.GeneralOpenAIRequest{Model: "gpt-5"}
	message := dto.Message{Role: "user"}
	message.SetMediaContent([]dto.MediaContent{{
		Type:         dto.ContentTypeText,
		Text:         "hello",
		CacheControl: []byte(`{"type":"ephemeral"}`),
	}})
	request.Messages = []dto.Message{message}

	responsesRequest, err := ChatCompletionsRequestToResponsesRequest(request)

	require.NoError(t, err)
	require.NotContains(t, string(responsesRequest.Input), "cache_control")
	require.JSONEq(t, `"in_memory"`, string(responsesRequest.PromptCacheRetention))
}
