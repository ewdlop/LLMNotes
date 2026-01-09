# Hugging Face VLM Models Analysis

This folder contains a comprehensive Excel workbook analyzing the most popular Vision-Language Models (VLMs) on Hugging Face.

## File

- **HuggingFace_VLM_Models.xlsx** - Complete VLM model analysis workbook

## Workbook Structure (6 Sheets)

| Sheet | Description |
|-------|-------------|
| **儀表板 (Dashboard)** | KPI metrics, Top 5 models summary, developer distribution, parameter size distribution |
| **下載量排名 (Downloads Ranking)** | 29 popular VLM models with full details (rank, name, developer, parameters, downloads, likes, links) |
| **官方推薦 (Official Picks)** | 6 Hugging Face recommended models with star ratings and use case suggestions |
| **模型分類 (Model Categories)** | Classification by model type (Any-to-Any, Reasoning, Small, etc.) and developer |
| **架構說明 (Architecture)** | VLM technical architecture explanation (fusion strategies, alignment methods) |
| **計算工具 (Calculator)** | Model selection calculator based on GPU memory and requirements |

## Key Findings

- **Qwen series** dominates the market, holding 4 of the top 5 download positions
- **moondream2** leads with 3.66M downloads (lightweight 2B model)
- **OCR-specialized models** (DeepSeek-OCR, HunyuanOCR) show strong demand
- **Small models** (<4B parameters) have the highest share, reflecting edge deployment trends

## Top 10 VLM Models by Downloads

| Rank | Model | Developer | Parameters | Downloads |
|------|-------|-----------|------------|-----------|
| 1 | moondream2 | vikhyatk | 2B | 3.66M |
| 2 | DeepSeek-OCR | deepseek-ai | 3B | 3.29M |
| 3 | Qwen3-VL-8B-Instruct | Qwen | 9B | 2.35M |
| 4 | Qwen2.5-VL-7B-Instruct | Qwen | 8B | 2.25M |
| 5 | Qwen2.5-VL-3B-Instruct | Qwen | 4B | 2.05M |
| 6 | openvla-7b | openvla | 8B | 1.79M |
| 7 | Qwen2-VL-2B-Instruct | Qwen | 2B | 1.54M |
| 8 | gemma-3-27b-it | Google | 27B | 1.42M |
| 9 | Qwen3-VL-30B-A3B-Instruct | Qwen | 31B | 1.25M |
| 10 | gemma-3-12b-it | Google | 12B | 1.2M |

## Model Categories

### By Type
- **Any-to-Any**: Qwen2.5-Omni, MiniCPM-o, Janus-Pro
- **Reasoning**: QVQ-72B, Kimi-VL-Thinking
- **Small/Efficient**: SmolVLM, moondream2, Florence-2
- **MoE**: Kimi-VL, Qwen3-VL-30B-A3B
- **OCR Specialized**: DeepSeek-OCR, HunyuanOCR, dots.ocr

### By Developer
- **Qwen (Alibaba)**: Qwen-VL, Qwen2-VL, Qwen2.5-VL, Qwen3-VL series
- **Google DeepMind**: Gemma-3, PaliGemma
- **Microsoft**: Florence-2, Phi-3.5-vision, Fara
- **OpenGVLab**: InternVL, InternVL2, InternVL3

## Data Sources

- [Hugging Face Models Hub](https://huggingface.co/models?pipeline_tag=image-text-to-text)
- [Hugging Face VLM Blog (May 2025)](https://huggingface.co/blog/vlms-2025)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

## Excel Features

The workbook follows best practices:
- ✓ A1 syntax for all formulas
- ✓ Cross-sheet references with proper `'Sheet Name'!A1` format
- ✓ Non-volatile functions (SUMIFS, INDEX, IFERROR, etc.)
- ✓ Error handling with IFERROR wrappers
- ✓ Freeze panes for easy navigation
- ✓ Data bars and color scales for visualization
- ✓ Model selection calculator for recommendations
- ✓ Star rating visualization for official picks

---

*Data collected: January 2025*
