# Claude

Notes on Anthropic's conversational AI models and their ecosystem.

## Model evolution
- **Claude 1.x (2023)**: Initial release with strong natural conversation capabilities and safety alignment.
- **Claude 2.x (2023)**: Extended context window (100K+ tokens), improved reasoning, and better instruction-following.
- **Claude 3 Family (2024)**: Three-tier model family with varying capability/cost trade-offs:
  - **Claude 3 Opus**: Highest capability, strongest reasoning and complex task handling.
  - **Claude 3 Sonnet**: Balanced performance and speed for enterprise workloads.
  - **Claude 3 Haiku**: Fastest, most compact for high-volume use cases.
- **Claude 3.5 Sonnet (2024)**: Enhanced coding abilities, better tool use, improved vision capabilities, and artifact generation.
- **Claude 3.5 Haiku (2024)**: Fastest model with improved intelligence while maintaining speed advantages.

## Typical usage patterns
- **System prompts**: Use the system parameter to set context, role, and behavior guidelines. Keep system prompts concise and focused.
- **Extended thinking**: Claude excels at chain-of-thought reasoning when explicitly prompted to think step-by-step.
- **Document analysis**: Strong performance on long-document understanding with context windows up to 200K tokens.
- **Structured outputs**: Supports JSON mode and XML-style tags for formatted responses. Prefers explicit structure markers.
- **Tool use**: Native function calling with comprehensive schemas. Models can plan multi-step tool sequences.
- **Vision capabilities**: Claude 3.x models support image inputs for document analysis, UI understanding, and visual reasoning.
- **Artifacts**: Claude 3.5+ can generate standalone content (code, documents, diagrams) in a separate artifact container.

## Safety and reliability considerations
- Constitutional AI alignment: Models trained with reinforcement learning from human feedback (RLHF) and Constitutional AI methods.
- Built-in harmlessness training: Designed to refuse harmful requests while maintaining helpfulness.
- Reduced hallucination: Strong emphasis on accuracy and citing limitations when uncertain.
- Privacy-focused: Anthropic commits to not training on customer data without explicit permission.
- Content moderation: Apply additional filtering for sensitive applications. Log interactions for compliance needs.
- Rate limiting: Implement application-layer rate limits and retry logic with exponential backoff.

## Evaluation ideas
- **Automated**: regression tests for JSON formatting, tool call accuracy, and prompt-response consistency.
- **Human evaluation**: assess reasoning quality, safety refusals, factual accuracy, and adherence to guidelines.
- **Benchmarks**: track performance on domain-specific tasks, coding challenges, and long-context understanding.
- **Metrics**: monitor latency, token usage, API error rates, and user satisfaction scores.

## Deployment references
- **Anthropic API**: Access via direct API with models: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`.
- **AWS Bedrock**: Claude models available through Amazon Bedrock for AWS-integrated deployments.
- **Google Cloud Vertex AI**: Claude 3 models available through Vertex AI Model Garden.
- **Context windows**: 200K token context for Claude 3 models, with extended context support in development.
- **Multimodality**: Vision support in Claude 3.x for image analysis, PDF processing, and screenshot understanding.
- **Pricing**: Tiered pricing per input/output token, with variations across model tiers. Vision inputs charged separately.

## Quick start (API)
1) Create an API key in the Anthropic Console and set it as `ANTHROPIC_API_KEY`.
2) Install SDK: `pip install anthropic`.
3) Minimal call:

```python
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)
print(message.content[0].text)
```

4) For tool use, define `tools` with JSON schemas and handle `tool_use` content blocks in responses.

## Industry standards and best practices
- **Prompt engineering**: Use clear instructions, examples, and explicit output formatting. Claude responds well to XML tags for structure.
- **Context management**: For long conversations, periodically summarize or trim context. Use Claude's extended context for document analysis.
- **Temperature settings**: Lower (0.0-0.3) for consistent/deterministic outputs, higher (0.7-1.0) for creative tasks.
- **Streaming**: Enable streaming for real-time response generation in interactive applications.
- **Error handling**: Implement retry logic with exponential backoff. Handle rate limits (429) and overload errors (529) gracefully.
- **Monitoring**: Track token usage, latency percentiles, error rates, and model performance metrics.

## Key differentiators
- **Extended context**: Industry-leading context windows enable analysis of entire codebases, books, or document collections.
- **Reasoning depth**: Strong performance on complex, multi-step reasoning tasks requiring careful analysis.
- **Safety alignment**: Constitutional AI methods provide robust safety without excessive refusals.
- **Honest uncertainty**: Models readily acknowledge limitations and uncertainties in their knowledge.
- **Code generation**: Particularly strong at writing, explaining, and debugging code across multiple languages.
- **Document understanding**: Excellent at extracting information and answering questions from long documents.

## Resources
- **Documentation**: https://docs.anthropic.com/
- **API Reference**: https://docs.anthropic.com/claude/reference/
- **Prompt Library**: https://docs.anthropic.com/claude/prompt-library
- **Model comparison**: https://www.anthropic.com/claude
