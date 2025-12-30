# ChatGPT

Notes on OpenAI's conversational GPT models and their ecosystem.

## Model evolution
- **GPT-3.5 (ChatGPT, 2022)**: First broadly available chat-tuned GPT release with instruction-following and dialogue memory.
- **GPT-4 (2023)**: Added stronger reasoning, tool-usage readiness, and more reliable safety behaviors.
- **GPT-4 Turbo / GPT-4o (2023-2024)**: Cheaper, faster variants with longer context and multimodal inputs/outputs (text, vision, and audio for GPT-4o).
- **GPT-4.1 / GPT-4.1 Mini (2024-2025)**: Reinforced tool-use capabilities, lower latency, and improved general reliability across reasoning and coding tasks.

## Typical usage patterns
- **System and user messages**: Use a system prompt to set behavior and provide user turns to guide the task. Add developer messages when coordinating tool calls or role guidance.
- **Few-shot steering**: Provide concise exemplars for formatting, safety boundaries, and style. Keep examples short to conserve context.
- **Structured outputs**: Prefer JSON schema or explicit field-level instructions when the response is machine-consumed.
- **Tool use**: Pair with functions/tools for retrieval, calculations, or mutations. Provide a JSON schema with required/optional properties and concise, deterministic descriptions.
- **Context control**: Trim irrelevant conversation history, summarize earlier turns, and include the task objective in the latest prompt to avoid drift.

## Safety and reliability considerations
- Apply content filters and rate limiting at the application layer.
- Log prompts, tool calls, and responses for auditing. Avoid persisting sensitive user data longer than necessary.
- Provide explicit refusal policies and escalation paths for unsupported or unsafe requests.
- When hallucination risk is high, ask the model to cite sources, return uncertainty estimates, or propose verification steps.

## Evaluation ideas
- **Automated**: regression suites with expectation checks for style/formatting, unit tests for tool schemas, and golden prompts for critical flows.
- **Human-in-the-loop**: spot checks of safety refusals, grounding quality, adherence to company policies, and UX consistency.
- **Metrics**: track latency, token usage, conversation success rate, and safety violation rates.

## Deployment references
- **OpenAI Platform API**: `gpt-4o`, `gpt-4.1`, and mini variants (`gpt-4o-mini`, `gpt-4.1-mini`) are the current chat-grade models.
- **Context windows**: Models provide long contexts (tens to hundreds of k tokens). Verify limits per model/region before production rollout.
- **Multimodality**: GPT-4o/4o-mini support image and audio inputs/outputs; GPT-4.1 adds stronger agentic tooling but text I/O only at launch.
- **Pricing**: Expect tiered pricing per input/output token and possible discounts for batches or larger commitments.

## Quick start (API)
1) Create an API key in the OpenAI dashboard and load it as `OPENAI_API_KEY`.
2) Install SDK: `pip install openai`.
3) Minimal call:

```python
from openai import OpenAI

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello"}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

4) For function calling, define `tools` with JSON schemas and inspect `resp.choices[0].message.tool_calls`.
