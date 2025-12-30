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
- **Constitutional AI alignment**: Models trained with reinforcement learning from human feedback (RLHF) and Constitutional AI methods for robust safety.
- **Built-in harmlessness training**: Designed to refuse harmful requests while maintaining helpfulness across diverse use cases.
- **Reduced hallucination**: Strong emphasis on accuracy and citing limitations when uncertain about information.
- **Privacy-focused**: Anthropic commits to not training on customer data without explicit permission, supporting enterprise privacy needs.
- **Content moderation**: Additional filtering recommended for sensitive applications. Log interactions for compliance and audit requirements.
- **Rate limiting**: Application-layer rate limits and retry logic with exponential backoff advised for production resilience.

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
## Overview

Claude is a family of large language models developed by Anthropic, an AI safety company founded in 2021 by former OpenAI researchers including Dario Amodei and Daniela Amodei. Claude is designed with a strong emphasis on safety, helpfulness, and harmlessness, using Constitutional AI (CAI) training methods.

## Model Evolution

### Claude 1.x (2023 Q1)
- **Claude 1.0**: Initial release
- **Claude 1.2**: Improved performance
- **Claude 1.3**: Enhanced capabilities
- Context window: 9,000 tokens (later expanded to 100K)

### Claude 2.x (2023 Q2-Q3)
- **Claude 2.0**: Major upgrade with improved reasoning
- **Claude 2.1**: Extended context window (200K tokens), reduced hallucination rates
- Context window: 100K → 200K tokens
- Improved accuracy and lower rates of model hallucination

### Claude 3 Family (2024 Q1)
Revolutionary multimodal models with vision capabilities:

- **Claude 3 Opus**: Most capable model, superior performance on complex tasks
  - Context window: 200K tokens
  - Best for: Research, strategic analysis, complex problem-solving

- **Claude 3 Sonnet**: Balanced performance and speed
  - Context window: 200K tokens
  - Best for: Data processing, sales automation, time-sensitive tasks

- **Claude 3 Haiku**: Fastest and most compact model
  - Context window: 200K tokens
  - Best for: Customer support, content moderation, cost-effective tasks

### Claude 3.5 Family (2024 Q2-Q4)
Enhanced versions with improved capabilities:

- **Claude 3.5 Sonnet**: Significant upgrade to Sonnet
  - Released June 2024, updated October 2024
  - Superior coding capabilities
  - Computer use (beta): Can interact with computer interfaces
  - Context window: 200K tokens
  - Best for: Coding, agentic tasks, visual processing

- **Claude 3.5 Haiku**: Improved speed and performance
  - Released November 2024
  - Matches Claude 3 Opus on many intelligence benchmarks
  - Context window: 200K tokens
  - Best for: Fast, intelligent tasks at lower cost

### Claude 4 Family (2024 Q4-2025)
Latest generation with unprecedented capabilities:

- **Claude 4 Opus**: Most advanced model (model ID: claude-opus-4-5-20251101)
  - Released November 2024
  - State-of-the-art performance across all benchmarks
  - Enhanced reasoning and analysis capabilities

- **Claude 4.5 Sonnet**: Currently powering Claude Code (claude-sonnet-4-5-20250929)
  - Released late 2024/early 2025
  - Optimized for coding and technical tasks
  - Extended context handling
  - Advanced tool use and function calling

## Key Capabilities

### Multimodal Understanding
- **Vision**: Process and analyze images (PNG, JPG, GIF, WebP)
- **PDF Processing**: Extract and understand content from PDF documents
- **Charts and Diagrams**: Interpret data visualizations
- **Screenshots**: Analyze UI/UX elements

### Advanced Reasoning
- **Complex Problem Solving**: Mathematical proofs, scientific analysis
- **Strategic Thinking**: Business analysis, planning, decision support
- **Logical Deduction**: Multi-step reasoning chains
- **Code Understanding**: Deep comprehension of programming concepts

### Long Context Processing
- **200K Token Context**: Entire codebases, lengthy documents, books
- **Context Retention**: Maintains coherence across long conversations
- **Needle-in-Haystack**: >99% accuracy finding information in long contexts

### Code Generation
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- **Code Review**: Identify bugs, suggest improvements, explain code
- **Refactoring**: Modernize and optimize existing code
- **Testing**: Generate comprehensive test suites
- **Documentation**: Automatic comment and documentation generation

### Tool Use and Function Calling
- **API Integration**: Structured function calling
- **External Tools**: Calculator, search, databases, custom APIs
- **Agentic Workflows**: Multi-step task execution
- **Computer Use (Beta)**: Control computer interfaces (3.5 Sonnet)

### Multilingual Support
- Strong performance in multiple languages including:
  - English, Spanish, French, German, Italian, Portuguese
  - Chinese (Simplified & Traditional), Japanese, Korean
  - And many more

## Constitutional AI and Safety

Claude is trained using Constitutional AI (CAI), a novel approach that:

1. **Principle-Based Training**: Models learn from a set of ethical principles
2. **Self-Critique**: Models evaluate their own outputs against principles
3. **Harmlessness**: Reduces potential for harmful outputs
4. **Transparency**: Clear reasoning about ethical decisions

### Safety Features
- Reduced hallucination rates
- Refusal to engage in harmful activities
- Awareness of limitations
- Factual grounding
- Bias mitigation

## Use Cases

### Software Development
- Code generation and completion
- Bug detection and fixing
- Code review and optimization
- Technical documentation
- API integration

### Research and Analysis
- Literature review and synthesis
- Data analysis and interpretation
- Scientific writing
- Hypothesis generation
- Peer review assistance

### Business Applications
- Customer support automation
- Document processing and summarization
- Market research and analysis
- Content creation and editing
- Strategic planning

### Creative Work
- Writing assistance (articles, stories, scripts)
- Brainstorming and ideation
- Translation and localization
- Educational content creation

### Specialized Tasks
- Legal document analysis
- Medical literature review (not medical advice)
- Financial analysis
- Academic tutoring

## API Usage

### Authentication
```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-api-key-here"
)
```

### Basic Message
```python
message = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)

print(message.content[0].text)
```

### With System Prompt
```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048,
    system="You are a helpful coding assistant specializing in Python.",
    messages=[
        {"role": "user", "content": "Write a function to implement binary search"}
    ]
)
```

### Vision Capabilities
```python
import base64

with open("image.jpg", "rb") as image_file:
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ],
        }
    ],
)
```

### Tool Use
```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }
    }
]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
)
```

### Streaming Responses
```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a short poem about AI"}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Technical Specifications

### Model Comparison

| Model | Release Date | Context Window | Best For | Relative Cost |
|-------|-------------|----------------|----------|---------------|
| Claude Opus 4.5 | Nov 2024 | 200K | Most complex tasks | Highest |
| Claude Sonnet 4.5 | Dec 2024 | 200K | Coding, balanced tasks | High |
| Claude 3.5 Sonnet | Oct 2024 | 200K | Agentic tasks, vision | Medium-High |
| Claude 3.5 Haiku | Nov 2024 | 200K | Fast, intelligent tasks | Low |
| Claude 3 Opus | Mar 2024 | 200K | Complex analysis | High |
| Claude 3 Sonnet | Mar 2024 | 200K | Balanced workloads | Medium |
| Claude 3 Haiku | Mar 2024 | 200K | Speed, efficiency | Lowest |

### Token Limits
- **Input**: Up to 200,000 tokens (approximately 150,000 words or 500 pages)
- **Output**: Up to 8,192 tokens for most models
- **Extended output**: Available for specific use cases

### Performance Benchmarks

**Claude 3.5 Sonnet** (October 2024):
- GPQA (Graduate-Level Reasoning): 59.4%
- MMLU (General Knowledge): 88.7%
- HumanEval (Coding): 93.7%
- MATH (Problem Solving): 78.3%
- MMMU (Multimodal Understanding): 68.3%

**Claude 3 Opus**:
- Undergraduate-level expert knowledge (MMLU): 86.8%
- Graduate-level reasoning (GPQA): 50.4%
- Math problem-solving (MATH): 60.1%
- Code generation (HumanEval): 84.9%

## Best Practices

### Prompt Engineering

1. **Be Clear and Specific**
   - Provide detailed instructions
   - Specify desired format and structure
   - Include examples when helpful

2. **Use System Prompts**
   - Define role and behavior
   - Set context and constraints
   - Specify tone and style

3. **Structure Complex Tasks**
   - Break down into steps
   - Use XML tags for organization
   - Provide clear delimiters

4. **Leverage Long Context**
   - Include full documentation
   - Provide entire codebase context
   - Reference specific sections

### Example: Structured Prompt
```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    system="You are an expert Python developer focused on clean, efficient code.",
    messages=[
        {
            "role": "user",
            "content": """Please help me with the following task:

<task>
Create a Python class for managing a simple todo list
</task>

<requirements>
1. Add, remove, and list todos
2. Mark todos as complete
3. Save/load from JSON file
4. Include error handling
5. Add type hints
6. Write docstrings
</requirements>

<output_format>
Provide:
1. Complete code with comments
2. Usage example
3. Brief explanation of design choices
</output_format>"""
        }
    ]
)
```

### Safety and Reliability

1. **Validate Outputs**: Always review generated code and content
2. **Test Thoroughly**: Don't deploy generated code without testing
3. **Check for Bias**: Review outputs for potential biases
4. **Handle Errors**: Implement proper error handling in applications
5. **Monitor Usage**: Track API usage and costs
6. **Rate Limiting**: Implement appropriate rate limiting

### Cost Optimization

1. **Choose the Right Model**
   - Use Haiku for simple, high-volume tasks
   - Use Sonnet for balanced workloads
   - Reserve Opus for complex reasoning

2. **Optimize Context**
   - Only include necessary context
   - Use prompt caching for repeated content
   - Summarize long documents when possible

3. **Stream Responses**
   - Better user experience
   - Early termination if needed

4. **Batch Processing**
   - Group similar requests
   - Use async for parallel processing

## Claude Code

**Claude Code** is Anthropic's official CLI tool that provides an interactive development environment powered by Claude. It features:

- Interactive coding assistance
- File operations and code editing
- Git integration
- Web search and fetch capabilities
- Background task execution
- Tool use and extensibility

Currently powered by **Claude Sonnet 4.5** (claude-sonnet-4-5-20250929).

## Resources

### Official Links
- **Website**: https://www.anthropic.com
- **API Documentation**: https://docs.anthropic.com
- **Console**: https://console.anthropic.com
- **Discord**: https://discord.gg/anthropic
- **GitHub**: https://github.com/anthropics

### Research Papers
- "Constitutional AI: Harmlessness from AI Feedback" (2022)
- "Claude 3 Model Card" (2024)
- "Challenges in Evaluating AI Systems" (2024)

### Developer Tools
- **Python SDK**: anthropic-sdk-python
- **TypeScript SDK**: anthropic-sdk-typescript
- **Claude Code**: Official CLI tool
- **Prompt Library**: Example prompts and best practices

## Comparison with Other LLMs

### vs ChatGPT/GPT-4
- **Strengths**: Longer context (200K vs 128K), stronger safety focus, better at following complex instructions
- **Differences**: Constitutional AI vs RLHF, different API structure, different pricing model

### vs Google Gemini
- **Strengths**: Better code generation, stronger reasoning, more transparent safety approach
- **Similarities**: Both multimodal, long context, tool use capabilities

### vs Open Source Models
- **Advantages**: Stronger performance, better support, managed infrastructure
- **Trade-offs**: Proprietary vs open, cost vs self-hosting, privacy considerations

## Future Developments

Anthropic continues to develop Claude with focus on:
- Enhanced reasoning capabilities
- Improved multimodal understanding
- Better tool use and agentic behaviors
- Increased safety and alignment
- Longer context windows
- More efficient inference
- Broader domain expertise

---

**Last Updated**: December 2024
**Current Flagship Model**: Claude Opus 4.5 (claude-opus-4-5-20251101)
**Knowledge Cutoff**: January 2025 (for current models)
