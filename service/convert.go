package service

import (
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/relay/channel/openrouter"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/reasonmap"
	"github.com/samber/lo"
)

type claudeContentSegment struct {
	kind string
	text string
}

const (
	thinkOpenTag  = "<think>"
	thinkCloseTag = "</think>"
)

func appendClaudeContentSegment(segments *[]claudeContentSegment, kind string, text string) {
	if text == "" {
		return
	}
	last := len(*segments) - 1
	if last >= 0 && (*segments)[last].kind == kind {
		(*segments)[last].text += text
		return
	}
	*segments = append(*segments, claudeContentSegment{kind: kind, text: text})
}

func splitThinkTaggedContent(content string) []claudeContentSegment {
	if content == "" {
		return nil
	}

	trimmed := strings.TrimLeft(content, " \t\r\n")
	if !strings.HasPrefix(trimmed, thinkOpenTag) {
		return []claudeContentSegment{{kind: relaycommon.LastMessageTypeText, text: content}}
	}

	segments := make([]claudeContentSegment, 0)
	leading := len(content) - len(trimmed)
	start := leading + len(thinkOpenTag)
	closeOffset := strings.Index(content[start:], thinkCloseTag)
	if closeOffset == -1 {
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, strings.TrimSpace(content[start:]))
		return segments
	}

	close := start + closeOffset
	appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, strings.TrimSpace(content[start:close]))
	appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeText, content[close+len(thinkCloseTag):])
	return segments
}

func longestSuffixPrefix(s string, prefix string) int {
	maxLen := len(prefix) - 1
	if len(s) < maxLen {
		maxLen = len(s)
	}
	for i := maxLen; i > 0; i-- {
		if strings.HasSuffix(s, prefix[:i]) {
			return i
		}
	}
	return 0
}

func splitThinkTaggedStreamContent(content string, state *relaycommon.ClaudeConvertInfo) []claudeContentSegment {
	if state == nil {
		return splitThinkTaggedContent(content)
	}

	data := state.ThinkTagBuffer + content
	state.ThinkTagBuffer = ""
	segments := make([]claudeContentSegment, 0)

	if state.InThinkTag {
		close := strings.Index(data, thinkCloseTag)
		if close == -1 {
			keep := longestSuffixPrefix(data, thinkCloseTag)
			appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, data[:len(data)-keep])
			state.ThinkTagBuffer = data[len(data)-keep:]
			return segments
		}
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, data[:close])
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeText, data[close+len(thinkCloseTag):])
		state.InThinkTag = false
		return segments
	}

	if state.LastMessagesType != relaycommon.LastMessageTypeNone {
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeText, data)
		return segments
	}

	trimmed := strings.TrimLeft(data, " \t\r\n")
	if trimmed == "" {
		state.ThinkTagBuffer = data
		return nil
	}
	if strings.HasPrefix(thinkOpenTag, trimmed) {
		state.ThinkTagBuffer = data
		return nil
	}
	if !strings.HasPrefix(trimmed, thinkOpenTag) {
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeText, data)
		return segments
	}

	start := len(data) - len(trimmed) + len(thinkOpenTag)
	close := strings.Index(data[start:], thinkCloseTag)
	if close == -1 {
		state.InThinkTag = true
		keep := longestSuffixPrefix(data[start:], thinkCloseTag)
		appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, data[start:len(data)-keep])
		state.ThinkTagBuffer = data[len(data)-keep:]
		return segments
	}

	close += start
	appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeThinking, data[start:close])
	appendClaudeContentSegment(&segments, relaycommon.LastMessageTypeText, data[close+len(thinkCloseTag):])
	return segments
}

func flushThinkTagBuffer(state *relaycommon.ClaudeConvertInfo) []claudeContentSegment {
	if state == nil || state.ThinkTagBuffer == "" {
		return nil
	}
	kind := relaycommon.LastMessageTypeText
	if state.InThinkTag {
		kind = relaycommon.LastMessageTypeThinking
	}
	segment := claudeContentSegment{kind: kind, text: state.ThinkTagBuffer}
	state.ThinkTagBuffer = ""
	return []claudeContentSegment{segment}
}

func parseClaudeTool(raw any) (*dto.Tool, bool) {
	switch tool := raw.(type) {
	case dto.Tool:
		return &tool, true
	case *dto.Tool:
		return tool, tool != nil
	case map[string]any:
		name := common.Interface2String(tool["name"])
		if name == "" {
			return nil, false
		}
		parsed := &dto.Tool{
			Name:        name,
			Description: common.Interface2String(tool["description"]),
		}
		if schema, err := common.Any2Type[map[string]any](tool["input_schema"]); err == nil {
			parsed.InputSchema = schema
		}
		return parsed, true
	default:
		parsed, err := common.Any2Type[dto.Tool](raw)
		if err != nil || parsed.Name == "" {
			return nil, false
		}
		return &parsed, true
	}
}

func findClaudeToolInputSchema(info *relaycommon.RelayInfo, toolName string) map[string]any {
	if info == nil || toolName == "" {
		return nil
	}
	claudeRequest, ok := info.Request.(*dto.ClaudeRequest)
	if !ok || claudeRequest == nil || claudeRequest.Tools == nil {
		return nil
	}

	matchTool := func(tool *dto.Tool) map[string]any {
		if tool == nil || tool.Name != toolName || tool.InputSchema == nil {
			return nil
		}
		return tool.InputSchema
	}

	switch tools := claudeRequest.Tools.(type) {
	case []dto.Tool:
		for i := range tools {
			if schema := matchTool(&tools[i]); schema != nil {
				return schema
			}
		}
	case []*dto.Tool:
		for _, tool := range tools {
			if schema := matchTool(tool); schema != nil {
				return schema
			}
		}
	case []any:
		for _, raw := range tools {
			if tool, ok := parseClaudeTool(raw); ok {
				if schema := matchTool(tool); schema != nil {
					return schema
				}
			}
		}
	default:
		parsedTools, err := common.Any2Type[[]dto.Tool](claudeRequest.Tools)
		if err != nil {
			return nil
		}
		for i := range parsedTools {
			if schema := matchTool(&parsedTools[i]); schema != nil {
				return schema
			}
		}
	}
	return nil
}

func claudeToolRequiredSet(schema map[string]any) map[string]bool {
	required := make(map[string]bool)
	switch values := schema["required"].(type) {
	case []string:
		for _, value := range values {
			required[value] = true
		}
	case []any:
		for _, value := range values {
			if s, ok := value.(string); ok {
				required[s] = true
			}
		}
	}
	return required
}

func claudeToolPropertiesSet(schema map[string]any) map[string]bool {
	props, err := common.Any2Type[map[string]any](schema["properties"])
	if err != nil {
		return nil
	}
	properties := make(map[string]bool, len(props))
	for key := range props {
		properties[key] = true
	}
	return properties
}

func sanitizeClaudeToolInput(input map[string]any, toolName string, info *relaycommon.RelayInfo) (map[string]any, bool) {
	if len(input) == 0 {
		return input, false
	}
	schema := findClaudeToolInputSchema(info, toolName)
	if schema == nil {
		return input, false
	}
	properties := claudeToolPropertiesSet(schema)
	if len(properties) == 0 {
		return input, false
	}
	required := claudeToolRequiredSet(schema)

	var sanitized map[string]any
	for key, value := range input {
		if required[key] || !properties[key] {
			continue
		}
		if s, ok := value.(string); ok && s == "" {
			if sanitized == nil {
				sanitized = make(map[string]any, len(input))
				for copyKey, copyValue := range input {
					sanitized[copyKey] = copyValue
				}
			}
			delete(sanitized, key)
		}
	}
	if sanitized == nil {
		return input, false
	}
	return sanitized, true
}

func sanitizeClaudeToolArguments(arguments string, toolName string, info *relaycommon.RelayInfo) string {
	if strings.TrimSpace(arguments) == "" {
		return arguments
	}
	var input map[string]any
	if err := common.Unmarshal([]byte(arguments), &input); err != nil {
		return arguments
	}
	sanitized, changed := sanitizeClaudeToolInput(input, toolName, info)
	if !changed {
		return arguments
	}
	encoded, err := common.Marshal(sanitized)
	if err != nil {
		return arguments
	}
	return string(encoded)
}

func openAIPromptCacheRetentionFromClaude(cacheControl []byte) []byte {
	if len(cacheControl) == 0 {
		return nil
	}
	return []byte(`"in_memory"`)
}

func applyClaudeCacheControlToOpenAIRequest(openAIRequest *dto.GeneralOpenAIRequest, cacheControl []byte) {
	if openAIRequest == nil || len(cacheControl) == 0 || len(openAIRequest.PromptCacheRetention) > 0 {
		return
	}
	openAIRequest.PromptCacheRetention = openAIPromptCacheRetentionFromClaude(cacheControl)
}

func ClaudeToOpenAIRequest(claudeRequest dto.ClaudeRequest, info *relaycommon.RelayInfo) (*dto.GeneralOpenAIRequest, error) {
	openAIRequest := dto.GeneralOpenAIRequest{
		Model:       claudeRequest.Model,
		Temperature: claudeRequest.Temperature,
	}
	if claudeRequest.MaxTokens != nil {
		openAIRequest.MaxTokens = lo.ToPtr(lo.FromPtr(claudeRequest.MaxTokens))
	}
	if claudeRequest.TopP != nil {
		openAIRequest.TopP = lo.ToPtr(lo.FromPtr(claudeRequest.TopP))
	}
	if claudeRequest.TopK != nil {
		openAIRequest.TopK = lo.ToPtr(lo.FromPtr(claudeRequest.TopK))
	}
	if claudeRequest.Stream != nil {
		openAIRequest.Stream = lo.ToPtr(lo.FromPtr(claudeRequest.Stream))
	}

	channelType := 0
	upstreamModelName := ""
	originModelName := ""
	if info != nil && info.ChannelMeta != nil {
		channelType = info.ChannelType
		upstreamModelName = info.UpstreamModelName
		originModelName = info.OriginModelName
	}

	isOpenRouter := channelType == constant.ChannelTypeOpenRouter
	isOpenRouterClaude := isOpenRouter && strings.HasPrefix(upstreamModelName, "anthropic/claude")
	if !isOpenRouterClaude {
		applyClaudeCacheControlToOpenAIRequest(&openAIRequest, claudeRequest.CacheControl)
	}

	if isOpenRouter {
		if effort := claudeRequest.GetEfforts(); effort != "" {
			effortBytes, _ := common.Marshal(effort)
			openAIRequest.Verbosity = effortBytes
		}
		if claudeRequest.Thinking != nil {
			var reasoning openrouter.RequestReasoning
			if claudeRequest.Thinking.Type == "enabled" {
				reasoning = openrouter.RequestReasoning{
					Enabled:   true,
					MaxTokens: claudeRequest.Thinking.GetBudgetTokens(),
				}
			} else if claudeRequest.Thinking.Type == "adaptive" {
				reasoning = openrouter.RequestReasoning{
					Enabled: true,
				}
			}
			reasoningJSON, err := common.Marshal(reasoning)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal reasoning: %w", err)
			}
			openAIRequest.Reasoning = reasoningJSON
		}
	} else {
		thinkingSuffix := "-thinking"
		if strings.HasSuffix(originModelName, thinkingSuffix) &&
			!strings.HasSuffix(openAIRequest.Model, thinkingSuffix) {
			openAIRequest.Model = openAIRequest.Model + thinkingSuffix
		}
	}

	// Convert stop sequences
	if len(claudeRequest.StopSequences) == 1 {
		openAIRequest.Stop = claudeRequest.StopSequences[0]
	} else if len(claudeRequest.StopSequences) > 1 {
		openAIRequest.Stop = claudeRequest.StopSequences
	}

	// Convert tools
	tools, _ := common.Any2Type[[]dto.Tool](claudeRequest.Tools)
	openAITools := make([]dto.ToolCallRequest, 0)
	for _, claudeTool := range tools {
		openAITool := dto.ToolCallRequest{
			Type: "function",
			Function: dto.FunctionRequest{
				Name:        claudeTool.Name,
				Description: claudeTool.Description,
				Parameters:  claudeTool.InputSchema,
			},
		}
		openAITools = append(openAITools, openAITool)
	}
	openAIRequest.Tools = openAITools

	// Convert messages
	openAIMessages := make([]dto.Message, 0)

	// Add system message if present
	if claudeRequest.System != nil {
		if claudeRequest.IsStringSystem() && claudeRequest.GetStringSystem() != "" {
			openAIMessage := dto.Message{
				Role: "system",
			}
			openAIMessage.SetStringContent(claudeRequest.GetStringSystem())
			openAIMessages = append(openAIMessages, openAIMessage)
		} else {
			systems := claudeRequest.ParseSystem()
			if len(systems) > 0 {
				openAIMessage := dto.Message{
					Role: "system",
				}
				isOpenRouterClaude := isOpenRouter && strings.HasPrefix(info.UpstreamModelName, "anthropic/claude")
				if isOpenRouterClaude {
					systemMediaMessages := make([]dto.MediaContent, 0, len(systems))
					for _, system := range systems {
						message := dto.MediaContent{
							Type:         "text",
							Text:         system.GetText(),
							CacheControl: system.CacheControl,
						}
						systemMediaMessages = append(systemMediaMessages, message)
					}
					openAIMessage.SetMediaContent(systemMediaMessages)
				} else {
					systemStr := ""
					for _, system := range systems {
						applyClaudeCacheControlToOpenAIRequest(&openAIRequest, system.CacheControl)
						if system.Text != nil {
							systemStr += *system.Text
						}
					}
					openAIMessage.SetStringContent(systemStr)
				}
				openAIMessages = append(openAIMessages, openAIMessage)
			}
		}
	}
	for _, claudeMessage := range claudeRequest.Messages {
		openAIMessage := dto.Message{
			Role: claudeMessage.Role,
		}

		//log.Printf("claudeMessage.Content: %v", claudeMessage.Content)
		if claudeMessage.IsStringContent() {
			openAIMessage.SetStringContent(claudeMessage.GetStringContent())
		} else {
			content, err := claudeMessage.ParseContent()
			if err != nil {
				return nil, err
			}
			contents := content
			var toolCalls []dto.ToolCallRequest
			mediaMessages := make([]dto.MediaContent, 0, len(contents))

			for _, mediaMsg := range contents {
				switch mediaMsg.Type {
				case "text", "input_text":
					message := dto.MediaContent{
						Type:         "text",
						Text:         mediaMsg.GetText(),
					}
					if isOpenRouterClaude {
						message.CacheControl = mediaMsg.CacheControl
					} else {
						applyClaudeCacheControlToOpenAIRequest(&openAIRequest, mediaMsg.CacheControl)
					}
					mediaMessages = append(mediaMessages, message)
				case "image":
					// Handle image conversion (base64 to URL or keep as is)
					imageData := fmt.Sprintf("data:%s;base64,%s", mediaMsg.Source.MediaType, mediaMsg.Source.Data)
					//textContent += fmt.Sprintf("[Image: %s]", imageData)
					mediaMessage := dto.MediaContent{
						Type:     "image_url",
						ImageUrl: &dto.MessageImageUrl{Url: imageData},
					}
					mediaMessages = append(mediaMessages, mediaMessage)
				case "tool_use":
					toolCall := dto.ToolCallRequest{
						ID:   mediaMsg.Id,
						Type: "function",
						Function: dto.FunctionRequest{
							Name:      mediaMsg.Name,
							Arguments: toJSONString(mediaMsg.Input),
						},
					}
					toolCalls = append(toolCalls, toolCall)
				case "tool_result":
					// Add tool result as a separate message
					toolName := mediaMsg.Name
					if toolName == "" {
						toolName = claudeRequest.SearchToolNameByToolCallId(mediaMsg.ToolUseId)
					}
					oaiToolMessage := dto.Message{
						Role:       "tool",
						Name:       &toolName,
						ToolCallId: mediaMsg.ToolUseId,
					}
					//oaiToolMessage.SetStringContent(*mediaMsg.GetMediaContent().Text)
					if mediaMsg.IsStringContent() {
						oaiToolMessage.SetStringContent(mediaMsg.GetStringContent())
					} else {
						mediaContents := mediaMsg.ParseMediaContent()
						encodeJson, _ := common.Marshal(mediaContents)
						oaiToolMessage.SetStringContent(string(encodeJson))
					}
					openAIMessages = append(openAIMessages, oaiToolMessage)
				}
			}

			if len(toolCalls) > 0 {
				openAIMessage.SetToolCalls(toolCalls)
			}

			if len(mediaMessages) > 0 && len(toolCalls) == 0 {
				openAIMessage.SetMediaContent(mediaMessages)
			}
		}
		if len(openAIMessage.ParseContent()) > 0 || len(openAIMessage.ToolCalls) > 0 {
			openAIMessages = append(openAIMessages, openAIMessage)
		}
	}

	openAIRequest.Messages = openAIMessages

	return &openAIRequest, nil
}

func generateStopBlock(index int) *dto.ClaudeResponse {
	return &dto.ClaudeResponse{
		Type:  "content_block_stop",
		Index: common.GetPointer[int](index),
	}
}

func buildClaudeUsageFromOpenAIUsage(oaiUsage *dto.Usage) *dto.ClaudeUsage {
	if oaiUsage == nil {
		return nil
	}
	cacheCreation5m, cacheCreation1h := NormalizeCacheCreationSplit(
		oaiUsage.PromptTokensDetails.CachedCreationTokens,
		oaiUsage.ClaudeCacheCreation5mTokens,
		oaiUsage.ClaudeCacheCreation1hTokens,
	)
	usage := &dto.ClaudeUsage{
		InputTokens:              oaiUsage.PromptTokens,
		OutputTokens:             oaiUsage.CompletionTokens,
		CacheCreationInputTokens: oaiUsage.PromptTokensDetails.CachedCreationTokens,
		CacheReadInputTokens:     oaiUsage.PromptTokensDetails.CachedTokens,
	}
	if cacheCreation5m > 0 || cacheCreation1h > 0 {
		usage.CacheCreation = &dto.ClaudeCacheCreationUsage{
			Ephemeral5mInputTokens: cacheCreation5m,
			Ephemeral1hInputTokens: cacheCreation1h,
		}
	}
	return usage
}

func NormalizeCacheCreationSplit(totalTokens int, tokens5m int, tokens1h int) (int, int) {
	remainder := lo.Max([]int{totalTokens - tokens5m - tokens1h, 0})
	return tokens5m + remainder, tokens1h
}

func StreamResponseOpenAI2Claude(openAIResponse *dto.ChatCompletionsStreamResponse, info *relaycommon.RelayInfo) []*dto.ClaudeResponse {
	if info.ClaudeConvertInfo.Done {
		return nil
	}

	var claudeResponses []*dto.ClaudeResponse

	ensureToolCallBuffers := func() {
		if info.ClaudeConvertInfo.ToolCallArgumentBuffers == nil {
			info.ClaudeConvertInfo.ToolCallArgumentBuffers = make(map[int]string)
		}
		if info.ClaudeConvertInfo.ToolCallNames == nil {
			info.ClaudeConvertInfo.ToolCallNames = make(map[int]string)
		}
	}
	setToolCallName := func(blockIndex int, name string) {
		if name == "" {
			return
		}
		ensureToolCallBuffers()
		info.ClaudeConvertInfo.ToolCallNames[blockIndex] = name
	}
	appendToolCallArguments := func(blockIndex int, arguments string) {
		if arguments == "" {
			return
		}
		ensureToolCallBuffers()
		info.ClaudeConvertInfo.ToolCallArgumentBuffers[blockIndex] += arguments
	}
	flushToolCallArguments := func() {
		if len(info.ClaudeConvertInfo.ToolCallArgumentBuffers) == 0 {
			return
		}
		base := info.ClaudeConvertInfo.ToolCallBaseIndex
		for offset := 0; offset <= info.ClaudeConvertInfo.ToolCallMaxIndexOffset; offset++ {
			blockIndex := base + offset
			arguments := info.ClaudeConvertInfo.ToolCallArgumentBuffers[blockIndex]
			if arguments == "" {
				continue
			}
			arguments = sanitizeClaudeToolArguments(arguments, info.ClaudeConvertInfo.ToolCallNames[blockIndex], info)
			idx := blockIndex
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Index: &idx,
				Type:  "content_block_delta",
				Delta: &dto.ClaudeMediaMessage{
					Type:        "input_json_delta",
					PartialJson: &arguments,
				},
			})
		}
		info.ClaudeConvertInfo.ToolCallArgumentBuffers = nil
		info.ClaudeConvertInfo.ToolCallNames = nil
	}

	var appendContentSegments func([]claudeContentSegment)

	stopOpenBlocks := func() {
		switch info.ClaudeConvertInfo.LastMessagesType {
		case relaycommon.LastMessageTypeText, relaycommon.LastMessageTypeThinking:
			appendContentSegments(flushThinkTagBuffer(info.ClaudeConvertInfo))
			claudeResponses = append(claudeResponses, generateStopBlock(info.ClaudeConvertInfo.Index))
		case relaycommon.LastMessageTypeTools:
			flushToolCallArguments()
			base := info.ClaudeConvertInfo.ToolCallBaseIndex
			for offset := 0; offset <= info.ClaudeConvertInfo.ToolCallMaxIndexOffset; offset++ {
				claudeResponses = append(claudeResponses, generateStopBlock(base+offset))
			}
		}
	}
	stopOpenBlocksAndAdvance := func() {
		if info.ClaudeConvertInfo.LastMessagesType == relaycommon.LastMessageTypeNone {
			return
		}
		stopOpenBlocks()
		switch info.ClaudeConvertInfo.LastMessagesType {
		case relaycommon.LastMessageTypeTools:
			info.ClaudeConvertInfo.Index = info.ClaudeConvertInfo.ToolCallBaseIndex + info.ClaudeConvertInfo.ToolCallMaxIndexOffset + 1
			info.ClaudeConvertInfo.ToolCallBaseIndex = 0
			info.ClaudeConvertInfo.ToolCallMaxIndexOffset = 0
		default:
			info.ClaudeConvertInfo.Index++
		}
		info.ClaudeConvertInfo.LastMessagesType = relaycommon.LastMessageTypeNone
	}
	appendContentSegment := func(segment claudeContentSegment) {
		if segment.text == "" {
			return
		}
		switch segment.kind {
		case relaycommon.LastMessageTypeThinking:
			if info.ClaudeConvertInfo.LastMessagesType != relaycommon.LastMessageTypeThinking {
				stopOpenBlocksAndAdvance()
				idx := info.ClaudeConvertInfo.Index
				claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
					Index: &idx,
					Type:  "content_block_start",
					ContentBlock: &dto.ClaudeMediaMessage{
						Type:     "thinking",
						Thinking: common.GetPointer[string](""),
					},
				})
			}
			info.ClaudeConvertInfo.LastMessagesType = relaycommon.LastMessageTypeThinking
			idx := info.ClaudeConvertInfo.Index
			thinking := segment.text
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Index: &idx,
				Type:  "content_block_delta",
				Delta: &dto.ClaudeMediaMessage{
					Type:     "thinking_delta",
					Thinking: &thinking,
				},
			})
		default:
			if info.ClaudeConvertInfo.LastMessagesType != relaycommon.LastMessageTypeText {
				stopOpenBlocksAndAdvance()
				idx := info.ClaudeConvertInfo.Index
				claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
					Index: &idx,
					Type:  "content_block_start",
					ContentBlock: &dto.ClaudeMediaMessage{
						Type: "text",
						Text: common.GetPointer[string](""),
					},
				})
			}
			info.ClaudeConvertInfo.LastMessagesType = relaycommon.LastMessageTypeText
			idx := info.ClaudeConvertInfo.Index
			text := segment.text
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Index: &idx,
				Type:  "content_block_delta",
				Delta: &dto.ClaudeMediaMessage{
					Type: "text_delta",
					Text: &text,
				},
			})
		}
	}
	appendContentSegments = func(segments []claudeContentSegment) {
		for _, segment := range segments {
			appendContentSegment(segment)
		}
	}
	if info.SendResponseCount == 1 {
		msg := &dto.ClaudeMediaMessage{
			Id:    openAIResponse.Id,
			Model: openAIResponse.Model,
			Type:  "message",
			Role:  "assistant",
			Usage: &dto.ClaudeUsage{
				InputTokens:  info.GetEstimatePromptTokens(),
				OutputTokens: 0,
			},
		}
		msg.SetContent(make([]any, 0))
		claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
			Type:    "message_start",
			Message: msg,
		})
		//claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
		//	Type: "ping",
		//})
		if openAIResponse.IsToolCall() {
			info.ClaudeConvertInfo.LastMessagesType = relaycommon.LastMessageTypeTools
			info.ClaudeConvertInfo.ToolCallBaseIndex = 0
			info.ClaudeConvertInfo.ToolCallMaxIndexOffset = 0
			var toolCall dto.ToolCallResponse
			if len(openAIResponse.Choices) > 0 && len(openAIResponse.Choices[0].Delta.ToolCalls) > 0 {
				toolCall = openAIResponse.Choices[0].Delta.ToolCalls[0]
			} else {
				first := openAIResponse.GetFirstToolCall()
				if first != nil {
					toolCall = *first
				} else {
					toolCall = dto.ToolCallResponse{}
				}
			}
			resp := &dto.ClaudeResponse{
				Type: "content_block_start",
				ContentBlock: &dto.ClaudeMediaMessage{
					Id:    toolCall.ID,
					Type:  "tool_use",
					Name:  toolCall.Function.Name,
					Input: map[string]interface{}{},
				},
			}
			resp.SetIndex(0)
			setToolCallName(0, toolCall.Function.Name)
			claudeResponses = append(claudeResponses, resp)
			appendToolCallArguments(0, toolCall.Function.Arguments)
		} else {

		}
		// 判断首个响应是否存在内容（非标准的 OpenAI 响应）
		if len(openAIResponse.Choices) > 0 {
			reasoning := openAIResponse.Choices[0].Delta.GetReasoningContent()
			content := openAIResponse.Choices[0].Delta.GetContentString()

			if reasoning != "" {
				appendContentSegment(claudeContentSegment{kind: relaycommon.LastMessageTypeThinking, text: reasoning})
			} else if content != "" {
				appendContentSegments(splitThinkTaggedStreamContent(content, info.ClaudeConvertInfo))
			}
		}

		// 如果首块就带 finish_reason，需要立即发送停止块
		if len(openAIResponse.Choices) > 0 && openAIResponse.Choices[0].FinishReason != nil && *openAIResponse.Choices[0].FinishReason != "" {
			info.FinishReason = *openAIResponse.Choices[0].FinishReason
			stopOpenBlocks()
			oaiUsage := openAIResponse.Usage
			if oaiUsage == nil {
				oaiUsage = info.ClaudeConvertInfo.Usage
			}
			if oaiUsage != nil {
				claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
					Type:  "message_delta",
					Usage: buildClaudeUsageFromOpenAIUsage(oaiUsage),
					Delta: &dto.ClaudeMediaMessage{
						StopReason: common.GetPointer[string](stopReasonOpenAI2Claude(info.FinishReason)),
					},
				})
			}
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Type: "message_stop",
			})
			info.ClaudeConvertInfo.Done = true
		}
		return claudeResponses
	}

	if len(openAIResponse.Choices) == 0 {
		// Some OpenAI-compatible upstreams end with a usage-only SSE chunk.
		oaiUsage := openAIResponse.Usage
		if oaiUsage == nil {
			oaiUsage = info.ClaudeConvertInfo.Usage
		}
		if oaiUsage != nil {
			stopOpenBlocks()
			stopReason := stopReasonOpenAI2Claude(info.FinishReason)
			if stopReason == "" {
				stopReason = "end_turn"
			}
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Type:  "message_delta",
				Usage: buildClaudeUsageFromOpenAIUsage(oaiUsage),
				Delta: &dto.ClaudeMediaMessage{
					StopReason: common.GetPointer[string](stopReason),
				},
			})
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Type: "message_stop",
			})
			info.ClaudeConvertInfo.Done = true
		}
		return claudeResponses
	} else {
		chosenChoice := openAIResponse.Choices[0]
		doneChunk := chosenChoice.FinishReason != nil && *chosenChoice.FinishReason != ""
		if doneChunk {
			info.FinishReason = *chosenChoice.FinishReason
			oaiUsage := openAIResponse.Usage
			if oaiUsage == nil {
				oaiUsage = info.ClaudeConvertInfo.Usage
				// Some upstreams emit finish_reason first, then send a final usage-only chunk.
				// Defer closing until usage is available so the final message_delta carries it.
				return claudeResponses
			}
		}

		var claudeResponse dto.ClaudeResponse
		var isEmpty bool
		claudeResponse.Type = "content_block_delta"
		if len(chosenChoice.Delta.ToolCalls) > 0 {
			toolCalls := chosenChoice.Delta.ToolCalls
			if info.ClaudeConvertInfo.LastMessagesType != relaycommon.LastMessageTypeTools {
				stopOpenBlocksAndAdvance()
				info.ClaudeConvertInfo.ToolCallBaseIndex = info.ClaudeConvertInfo.Index
				info.ClaudeConvertInfo.ToolCallMaxIndexOffset = 0
			}
			info.ClaudeConvertInfo.LastMessagesType = relaycommon.LastMessageTypeTools
			base := info.ClaudeConvertInfo.ToolCallBaseIndex
			maxOffset := info.ClaudeConvertInfo.ToolCallMaxIndexOffset

			for i, toolCall := range toolCalls {
				offset := 0
				if toolCall.Index != nil {
					offset = *toolCall.Index
				} else {
					offset = i
				}
				if offset > maxOffset {
					maxOffset = offset
				}
				blockIndex := base + offset

				idx := blockIndex
				if toolCall.Function.Name != "" {
					claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
						Index: &idx,
						Type:  "content_block_start",
						ContentBlock: &dto.ClaudeMediaMessage{
							Id:    toolCall.ID,
							Type:  "tool_use",
							Name:  toolCall.Function.Name,
							Input: map[string]interface{}{},
						},
					})
					setToolCallName(blockIndex, toolCall.Function.Name)
				}

				appendToolCallArguments(blockIndex, toolCall.Function.Arguments)
			}
			info.ClaudeConvertInfo.ToolCallMaxIndexOffset = maxOffset
			info.ClaudeConvertInfo.Index = base + maxOffset
		} else {
			reasoning := chosenChoice.Delta.GetReasoningContent()
			textContent := chosenChoice.Delta.GetContentString()
			if reasoning != "" {
				appendContentSegment(claudeContentSegment{kind: relaycommon.LastMessageTypeThinking, text: reasoning})
			} else if textContent != "" {
				appendContentSegments(splitThinkTaggedStreamContent(textContent, info.ClaudeConvertInfo))
			} else {
				isEmpty = true
			}
		}

		claudeResponse.Index = common.GetPointer[int](info.ClaudeConvertInfo.Index)
		if !isEmpty && claudeResponse.Delta != nil {
			claudeResponses = append(claudeResponses, &claudeResponse)
		}

		if doneChunk || info.ClaudeConvertInfo.Done {
			stopOpenBlocks()
			oaiUsage := openAIResponse.Usage
			if oaiUsage == nil {
				oaiUsage = info.ClaudeConvertInfo.Usage
			}
			if oaiUsage != nil {
				claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
					Type:  "message_delta",
					Usage: buildClaudeUsageFromOpenAIUsage(oaiUsage),
					Delta: &dto.ClaudeMediaMessage{
						StopReason: common.GetPointer[string](stopReasonOpenAI2Claude(info.FinishReason)),
					},
				})
			}
			claudeResponses = append(claudeResponses, &dto.ClaudeResponse{
				Type: "message_stop",
			})
			info.ClaudeConvertInfo.Done = true
			return claudeResponses
		}
	}

	return claudeResponses
}

func ResponseOpenAI2Claude(openAIResponse *dto.OpenAITextResponse, info *relaycommon.RelayInfo) *dto.ClaudeResponse {
	var stopReason string
	contents := make([]dto.ClaudeMediaMessage, 0)
	claudeResponse := &dto.ClaudeResponse{
		Id:    openAIResponse.Id,
		Type:  "message",
		Role:  "assistant",
		Model: openAIResponse.Model,
	}
	for _, choice := range openAIResponse.Choices {
		stopReason = stopReasonOpenAI2Claude(choice.FinishReason)
		if choice.FinishReason == "tool_calls" {
			for _, toolUse := range choice.Message.ParseToolCalls() {
				claudeContent := dto.ClaudeMediaMessage{}
				claudeContent.Type = "tool_use"
				claudeContent.Id = toolUse.ID
				claudeContent.Name = toolUse.Function.Name
				arguments := sanitizeClaudeToolArguments(toolUse.Function.Arguments, toolUse.Function.Name, info)
				var mapParams map[string]interface{}
				if err := common.Unmarshal([]byte(arguments), &mapParams); err == nil {
					claudeContent.Input = mapParams
				} else {
					claudeContent.Input = arguments
				}
				contents = append(contents, claudeContent)
			}
		} else {
			if reasoning := choice.Message.GetReasoningContent(); reasoning != "" {
				claudeContent := dto.ClaudeMediaMessage{Type: "thinking"}
				claudeContent.Thinking = &reasoning
				contents = append(contents, claudeContent)
			}
			for _, segment := range splitThinkTaggedContent(choice.Message.StringContent()) {
				claudeContent := dto.ClaudeMediaMessage{Type: "text"}
				if segment.kind == relaycommon.LastMessageTypeThinking {
					claudeContent.Type = "thinking"
					claudeContent.Thinking = &segment.text
				} else {
					claudeContent.SetText(segment.text)
				}
				contents = append(contents, claudeContent)
			}
		}
	}
	claudeResponse.Content = contents
	claudeResponse.StopReason = stopReason
	claudeResponse.Usage = buildClaudeUsageFromOpenAIUsage(&openAIResponse.Usage)

	return claudeResponse
}

func stopReasonOpenAI2Claude(reason string) string {
	return reasonmap.OpenAIFinishReasonToClaudeStopReason(reason)
}

func toJSONString(v interface{}) string {
	b, err := common.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func GeminiToOpenAIRequest(geminiRequest *dto.GeminiChatRequest, info *relaycommon.RelayInfo) (*dto.GeneralOpenAIRequest, error) {
	openaiRequest := &dto.GeneralOpenAIRequest{
		Model:  info.UpstreamModelName,
		Stream: lo.ToPtr(info.IsStream),
	}

	// 转换 messages
	var messages []dto.Message
	for _, content := range geminiRequest.Contents {
		message := dto.Message{
			Role: convertGeminiRoleToOpenAI(content.Role),
		}

		// 处理 parts
		var mediaContents []dto.MediaContent
		var toolCalls []dto.ToolCallRequest
		for _, part := range content.Parts {
			if part.Text != "" {
				mediaContent := dto.MediaContent{
					Type: "text",
					Text: part.Text,
				}
				mediaContents = append(mediaContents, mediaContent)
			} else if part.InlineData != nil {
				mediaContent := dto.MediaContent{
					Type: "image_url",
					ImageUrl: &dto.MessageImageUrl{
						Url:      fmt.Sprintf("data:%s;base64,%s", part.InlineData.MimeType, part.InlineData.Data),
						Detail:   "auto",
						MimeType: part.InlineData.MimeType,
					},
				}
				mediaContents = append(mediaContents, mediaContent)
			} else if part.FileData != nil {
				mediaContent := dto.MediaContent{
					Type: "image_url",
					ImageUrl: &dto.MessageImageUrl{
						Url:      part.FileData.FileUri,
						Detail:   "auto",
						MimeType: part.FileData.MimeType,
					},
				}
				mediaContents = append(mediaContents, mediaContent)
			} else if part.FunctionCall != nil {
				// 处理 Gemini 的工具调用
				toolCall := dto.ToolCallRequest{
					ID:   fmt.Sprintf("call_%d", len(toolCalls)+1), // 生成唯一ID
					Type: "function",
					Function: dto.FunctionRequest{
						Name:      part.FunctionCall.FunctionName,
						Arguments: toJSONString(part.FunctionCall.Arguments),
					},
				}
				toolCalls = append(toolCalls, toolCall)
			} else if part.FunctionResponse != nil {
				// 处理 Gemini 的工具响应，创建单独的 tool 消息
				toolMessage := dto.Message{
					Role:       "tool",
					ToolCallId: fmt.Sprintf("call_%d", len(toolCalls)), // 使用对应的调用ID
				}
				toolMessage.SetStringContent(toJSONString(part.FunctionResponse.Response))
				messages = append(messages, toolMessage)
			}
		}

		// 设置消息内容
		if len(toolCalls) > 0 {
			// 如果有工具调用，设置工具调用
			message.SetToolCalls(toolCalls)
		} else if len(mediaContents) == 1 && mediaContents[0].Type == "text" {
			// 如果只有一个文本内容，直接设置字符串
			message.Content = mediaContents[0].Text
		} else if len(mediaContents) > 0 {
			// 如果有多个内容或包含媒体，设置为数组
			message.SetMediaContent(mediaContents)
		}

		// 只有当消息有内容或工具调用时才添加
		if len(message.ParseContent()) > 0 || len(message.ToolCalls) > 0 {
			messages = append(messages, message)
		}
	}

	openaiRequest.Messages = messages

	if geminiRequest.GenerationConfig.Temperature != nil {
		openaiRequest.Temperature = geminiRequest.GenerationConfig.Temperature
	}
	if geminiRequest.GenerationConfig.TopP != nil && *geminiRequest.GenerationConfig.TopP > 0 {
		openaiRequest.TopP = lo.ToPtr(*geminiRequest.GenerationConfig.TopP)
	}
	if geminiRequest.GenerationConfig.TopK != nil && *geminiRequest.GenerationConfig.TopK > 0 {
		openaiRequest.TopK = lo.ToPtr(int(*geminiRequest.GenerationConfig.TopK))
	}
	if geminiRequest.GenerationConfig.MaxOutputTokens != nil && *geminiRequest.GenerationConfig.MaxOutputTokens > 0 {
		openaiRequest.MaxTokens = lo.ToPtr(*geminiRequest.GenerationConfig.MaxOutputTokens)
	}
	// gemini stop sequences 最多 5 个，openai stop 最多 4 个
	if len(geminiRequest.GenerationConfig.StopSequences) > 0 {
		openaiRequest.Stop = geminiRequest.GenerationConfig.StopSequences[:4]
	}
	if geminiRequest.GenerationConfig.CandidateCount != nil && *geminiRequest.GenerationConfig.CandidateCount > 0 {
		openaiRequest.N = lo.ToPtr(*geminiRequest.GenerationConfig.CandidateCount)
	}

	// 转换工具调用
	if len(geminiRequest.GetTools()) > 0 {
		var tools []dto.ToolCallRequest
		for _, tool := range geminiRequest.GetTools() {
			if tool.FunctionDeclarations != nil {
				functionDeclarations, err := common.Any2Type[[]dto.FunctionRequest](tool.FunctionDeclarations)
				if err != nil {
					common.SysError(fmt.Sprintf("failed to parse gemini function declarations: %v (type=%T)", err, tool.FunctionDeclarations))
					continue
				}
				for _, function := range functionDeclarations {
					openAITool := dto.ToolCallRequest{
						Type: "function",
						Function: dto.FunctionRequest{
							Name:        function.Name,
							Description: function.Description,
							Parameters:  function.Parameters,
						},
					}
					tools = append(tools, openAITool)
				}
			}
		}
		if len(tools) > 0 {
			openaiRequest.Tools = tools
		}
	}

	// gemini system instructions
	if geminiRequest.SystemInstructions != nil {
		// 将系统指令作为第一条消息插入
		systemMessage := dto.Message{
			Role:    "system",
			Content: extractTextFromGeminiParts(geminiRequest.SystemInstructions.Parts),
		}
		openaiRequest.Messages = append([]dto.Message{systemMessage}, openaiRequest.Messages...)
	}

	return openaiRequest, nil
}

func convertGeminiRoleToOpenAI(geminiRole string) string {
	switch geminiRole {
	case "user":
		return "user"
	case "model":
		return "assistant"
	case "function":
		return "function"
	default:
		return "user"
	}
}

func extractTextFromGeminiParts(parts []dto.GeminiPart) string {
	var texts []string
	for _, part := range parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// ResponseOpenAI2Gemini 将 OpenAI 响应转换为 Gemini 格式
func ResponseOpenAI2Gemini(openAIResponse *dto.OpenAITextResponse, info *relaycommon.RelayInfo) *dto.GeminiChatResponse {
	geminiResponse := &dto.GeminiChatResponse{
		Candidates: make([]dto.GeminiChatCandidate, 0, len(openAIResponse.Choices)),
		UsageMetadata: dto.GeminiUsageMetadata{
			PromptTokenCount:     openAIResponse.PromptTokens,
			CandidatesTokenCount: openAIResponse.CompletionTokens,
			TotalTokenCount:      openAIResponse.PromptTokens + openAIResponse.CompletionTokens,
		},
	}

	for _, choice := range openAIResponse.Choices {
		candidate := dto.GeminiChatCandidate{
			Index:         int64(choice.Index),
			SafetyRatings: []dto.GeminiChatSafetyRating{},
		}

		// 设置结束原因
		var finishReason string
		switch choice.FinishReason {
		case "stop":
			finishReason = "STOP"
		case "length":
			finishReason = "MAX_TOKENS"
		case "content_filter":
			finishReason = "SAFETY"
		case "tool_calls":
			finishReason = "STOP"
		default:
			finishReason = "STOP"
		}
		candidate.FinishReason = &finishReason

		// 转换消息内容
		content := dto.GeminiChatContent{
			Role:  "model",
			Parts: make([]dto.GeminiPart, 0),
		}

		// 处理工具调用
		toolCalls := choice.Message.ParseToolCalls()
		if len(toolCalls) > 0 {
			for _, toolCall := range toolCalls {
				// 解析参数
				var args map[string]interface{}
				if toolCall.Function.Arguments != "" {
					if err := common.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
						args = map[string]interface{}{"arguments": toolCall.Function.Arguments}
					}
				} else {
					args = make(map[string]interface{})
				}

				part := dto.GeminiPart{
					FunctionCall: &dto.FunctionCall{
						FunctionName: toolCall.Function.Name,
						Arguments:    args,
					},
				}
				content.Parts = append(content.Parts, part)
			}
		} else {
			// 处理文本内容
			textContent := choice.Message.StringContent()
			if textContent != "" {
				part := dto.GeminiPart{
					Text: textContent,
				}
				content.Parts = append(content.Parts, part)
			}
		}

		candidate.Content = content
		geminiResponse.Candidates = append(geminiResponse.Candidates, candidate)
	}

	return geminiResponse
}

// StreamResponseOpenAI2Gemini 将 OpenAI 流式响应转换为 Gemini 格式
func StreamResponseOpenAI2Gemini(openAIResponse *dto.ChatCompletionsStreamResponse, info *relaycommon.RelayInfo) *dto.GeminiChatResponse {
	// 检查是否有实际内容或结束标志
	hasContent := false
	hasFinishReason := false
	for _, choice := range openAIResponse.Choices {
		if len(choice.Delta.GetContentString()) > 0 || (choice.Delta.ToolCalls != nil && len(choice.Delta.ToolCalls) > 0) {
			hasContent = true
		}
		if choice.FinishReason != nil {
			hasFinishReason = true
		}
	}

	// 如果没有实际内容且没有结束标志，跳过。主要针对 openai 流响应开头的空数据
	if !hasContent && !hasFinishReason {
		return nil
	}

	geminiResponse := &dto.GeminiChatResponse{
		Candidates: make([]dto.GeminiChatCandidate, 0, len(openAIResponse.Choices)),
		UsageMetadata: dto.GeminiUsageMetadata{
			PromptTokenCount:     info.GetEstimatePromptTokens(),
			CandidatesTokenCount: 0, // 流式响应中可能没有完整的 usage 信息
			TotalTokenCount:      info.GetEstimatePromptTokens(),
		},
	}

	if openAIResponse.Usage != nil {
		geminiResponse.UsageMetadata.PromptTokenCount = openAIResponse.Usage.PromptTokens
		geminiResponse.UsageMetadata.CandidatesTokenCount = openAIResponse.Usage.CompletionTokens
		geminiResponse.UsageMetadata.TotalTokenCount = openAIResponse.Usage.TotalTokens
	}

	for _, choice := range openAIResponse.Choices {
		candidate := dto.GeminiChatCandidate{
			Index:         int64(choice.Index),
			SafetyRatings: []dto.GeminiChatSafetyRating{},
		}

		// 设置结束原因
		if choice.FinishReason != nil {
			var finishReason string
			switch *choice.FinishReason {
			case "stop":
				finishReason = "STOP"
			case "length":
				finishReason = "MAX_TOKENS"
			case "content_filter":
				finishReason = "SAFETY"
			case "tool_calls":
				finishReason = "STOP"
			default:
				finishReason = "STOP"
			}
			candidate.FinishReason = &finishReason
		}

		// 转换消息内容
		content := dto.GeminiChatContent{
			Role:  "model",
			Parts: make([]dto.GeminiPart, 0),
		}

		// 处理工具调用
		if choice.Delta.ToolCalls != nil {
			for _, toolCall := range choice.Delta.ToolCalls {
				// 解析参数
				var args map[string]interface{}
				if toolCall.Function.Arguments != "" {
					if err := common.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
						args = map[string]interface{}{"arguments": toolCall.Function.Arguments}
					}
				} else {
					args = make(map[string]interface{})
				}

				part := dto.GeminiPart{
					FunctionCall: &dto.FunctionCall{
						FunctionName: toolCall.Function.Name,
						Arguments:    args,
					},
				}
				content.Parts = append(content.Parts, part)
			}
		} else {
			// 处理文本内容
			textContent := choice.Delta.GetContentString()
			if textContent != "" {
				part := dto.GeminiPart{
					Text: textContent,
				}
				content.Parts = append(content.Parts, part)
			}
		}

		candidate.Content = content
		geminiResponse.Candidates = append(geminiResponse.Candidates, candidate)
	}

	return geminiResponse
}
