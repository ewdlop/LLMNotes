# Meta Llama

Notes on Meta's open-source large language models and their ecosystem.

## Model evolution
- **Llama 1 (February 2023)**: Initial release (7B-65B parameters) as open research models with restrictive licensing.
- **Llama 2 (July 2023)**: Commercially permissive open-source release (7B-70B) with strong chat variants and improved safety alignment.
- **Code Llama (August 2023)**: Specialized variant trained on code, supporting multiple programming languages with up to 100K context.
- **Llama 3 (April 2024)**: Major upgrade with 8B and 70B variants, 8K context, and significantly improved performance across benchmarks.
- **Llama 3.1 (July 2024)**: Extended context to 128K tokens, added 405B parameter model, improved multilingual and tool-use capabilities.
- **Llama 3.2 (September 2024)**: Added small models (1B, 3B) for edge deployment and multimodal vision models (11B, 90B).
- **Llama 3.3 (December 2024)**: 70B model matching Llama 3.1 405B performance at a fraction of the cost and compute.

## Key characteristics
- **Open source**: Released under permissive licenses allowing commercial use with minimal restrictions.
- **Multiple sizes**: Range from 1B (edge devices) to 405B (data center scale), enabling diverse deployment scenarios.
- **Strong base models**: Foundation models trained on 15T+ tokens of high-quality data, suitable for fine-tuning.
- **Instruction-tuned variants**: Chat models optimized for dialogue, instruction-following, and safety alignment.
- **Multilingual**: Llama 3.1+ supports 8 languages beyond English (German, French, Italian, Portuguese, Hindi, Spanish, Thai).
- **Tool use**: Native function calling and tool integration capabilities in Llama 3.1+.
- **Multimodal**: Llama 3.2 vision models support image understanding alongside text.

## Model family overview

### Llama 3.3 (December 2024)
- **70B**: Flagship model matching 405B performance with better efficiency and lower cost.

### Llama 3.2 (September 2024)
- **1B, 3B**: Lightweight models for mobile and edge deployment with 128K context.
- **11B, 90B Vision**: Multimodal models supporting text and image inputs.

### Llama 3.1 (July 2024)
- **8B**: Fast, efficient model for high-throughput applications.
- **70B**: Balanced performance/cost for production deployments.
- **405B**: Largest open model rivaling GPT-4 class performance.

## Deployment options
- **Direct inference**: Download model weights from Hugging Face or Meta and run locally or on cloud infrastructure.
- **Inference providers**: Use managed APIs from Together AI, Replicate, Groq, Fireworks AI, Anyscale, and others.
- **Cloud platforms**: Deploy via AWS Bedrock, Azure AI, Google Cloud Vertex AI, or OCI.
- **Local deployment**: Run smaller models (1B-13B) on consumer hardware using llama.cpp, Ollama, or LM Studio.
- **Fine-tuning platforms**: Customize models using Hugging Face, Modal, Databricks, or Weights & Biases.

## Typical usage patterns
- **Custom fine-tuning**: Start with base or instruct models and fine-tune for domain-specific tasks.
- **RAG applications**: Use as embedding models or generation models in retrieval-augmented generation pipelines.
- **On-premise deployment**: Host privately for data sensitivity, compliance, or air-gapped environments.
- **Research and experimentation**: Explore model architectures, training techniques, and alignment methods.
- **Cost optimization**: Replace expensive API calls with self-hosted Llama models for high-volume use cases.
- **Edge AI**: Deploy 1B-3B models on mobile devices, IoT, or edge servers for low-latency inference.

## Technical specifications (Llama 3.1)
- **Architecture**: Decoder-only transformer with grouped query attention (GQA).
- **Context length**: 128K tokens (8K for Llama 3.0).
- **Vocabulary**: 128K tokens using byte-pair encoding (BPE).
- **Training data**: 15T+ tokens across diverse domains, languages, and modalities.
- **Compute**: 405B model trained on 16K H100 GPUs over several months.
- **Quantization**: Supports FP16, BF16, INT8, and INT4 quantization for efficiency.

## Performance benchmarks
- **Llama 3.3 70B**: Matches or exceeds Llama 3.1 405B on most benchmarks while being 5x smaller.
- **Llama 3.1 405B**: Competitive with GPT-4, Claude 3.5 Sonnet on many tasks.
- **Code generation**: Strong performance on HumanEval, MBPP coding benchmarks.
- **Mathematics**: Improved reasoning on MATH and GSM8K problem sets.
- **Multilingual**: Competitive performance across 8 supported languages.
- **Long context**: Maintains strong performance across full 128K context window.

## Fine-tuning and customization
- **Base models**: Best starting point for task-specific fine-tuning with your own data.
- **Instruct models**: Pre-aligned for instruction-following; suitable for domain adaptation with minimal data.
- **PEFT methods**: Use LoRA, QLoRA, or prefix tuning for parameter-efficient fine-tuning.
- **Full fine-tuning**: Customize all parameters for maximum performance on specialized tasks.
- **RLHF**: Apply reinforcement learning from human feedback for alignment and safety.
- **Synthetic data**: Generate training data using larger models to improve smaller Llama variants.

## Safety and responsible AI
- **Llama Guard**: Specialized safety model for content moderation and prompt/response filtering.
- **Prompt Guard**: Protection against prompt injection and jailbreaking attempts.
- **Code Shield**: Security scanning for code generation outputs to detect vulnerabilities.
- **Red teaming**: Extensive adversarial testing for safety, fairness, and robustness.
- **Responsible use guide**: Meta provides comprehensive guidelines for ethical deployment.
- **Community reporting**: Open channels for reporting safety issues and model vulnerabilities.

## Quick start (local deployment with Ollama)
1) Install Ollama: Visit https://ollama.ai
2) Download and run model:

```bash
# Install and run Llama 3.2 (3B)
ollama run llama3.2

# Or larger variant
ollama run llama3.3

# Pull specific model
ollama pull llama3.1:70b
```

## Quick start (Python with Hugging Face)
1) Install dependencies: `pip install transformers torch accelerate`
2) Basic inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum entanglement."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

## Quick start (API via Together AI)
1) Sign up at https://together.ai and get API key
2) Install SDK: `pip install together`
3) Make API call:

```python
from together import Together

client = Together(api_key="your-api-key")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[
        {"role": "user", "content": "Write a Python function to merge sort a list."}
    ],
)

print(response.choices[0].message.content)
```

## Cost optimization strategies
- **Model selection**: Use smallest model that meets accuracy requirements (1B/3B for simple tasks, 70B+ for complex reasoning).
- **Quantization**: Apply INT8 or INT4 quantization to reduce memory and compute by 2-4x with minimal quality loss.
- **Batch processing**: Maximize GPU utilization by batching requests when real-time latency isn't critical.
- **Inference optimization**: Use vLLM, TGI (Text Generation Inference), or TensorRT-LLM for 2-5x speedup.
- **Edge deployment**: Move 1B-3B models to edge devices to eliminate cloud inference costs.
- **Caching**: Cache common prompts and responses to avoid redundant inference.

## Industry impact and ecosystem
- **Democratizing AI**: Provides access to state-of-the-art models without dependency on closed APIs.
- **Research acceleration**: Enables academic and independent research with frontier model capabilities.
- **Innovation platform**: Spawned thousands of fine-tuned variants, tools, and applications.
- **Economic impact**: Reduces inference costs for businesses by enabling self-hosting.
- **Geopolitical**: Challenges US tech concentration by providing globally accessible AI capabilities.
- **Education**: Enables students and educators to learn about LLMs with full model access.

## Licensing
- **Llama 3.x Community License**: Permissive open-source license allowing commercial use.
- **Restrictions**: Additional terms apply for services with >700M monthly active users.
- **Attribution**: Required to acknowledge Meta Llama in derivative works.
- **Acceptable use policy**: Prohibits harmful uses outlined in responsible use guide.

## Community and ecosystem
- **Hugging Face Hub**: Central repository for models, fine-tunes, and datasets.
- **Llama Recipes**: Official Meta repository with training scripts and examples.
- **Community fine-tunes**: Thousands of specialized variants for specific domains and languages.
- **Inference frameworks**: llama.cpp, Ollama, vLLM, TGI, and many others.
- **Deployment platforms**: Integration with major cloud providers and AI platforms.

## Resources
- **Official site**: https://llama.meta.com/
- **Model downloads**: https://huggingface.co/meta-llama
- **Documentation**: https://llama.meta.com/docs/
- **GitHub recipes**: https://github.com/meta-llama/llama-recipes
- **Research papers**: https://ai.meta.com/research/publications/
- **Community forum**: https://discuss.huggingface.co/c/llama/
