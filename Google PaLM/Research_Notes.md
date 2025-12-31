# Google PaLM Research Notes

## Overview
Google PaLM (Pathways Language Model) represents a significant milestone in large language model development, utilizing the Pathways architecture to enable highly efficient training across thousands of TPU chips. While the API has been decommissioned in favor of Gemini, PaLM 2 remains a technologically significant model.

## Technical Architecture
- **Pathways System**: A single model that can generalize across domains and tasks while being highly efficient. It orchestrates distributed computation for accelerators.
- **Scale**: PaLM was scaled to 540 billion parameters.
- **Training Infrastructure**: Trained on up to 6144 TPU v4 chips using the Pathways system.
- **Efficiency**: Achieved 57.8% hardware FLOPs utilization.
- **Model Structure**: Dense decoder-only Transformer model.

## PaLM 2
PaLM 2 is the successor, optimized for ease of use and available in four sizes:
- **Gecko**: Mobile-friendly, fast, suitable for on-device applications.
- **Otter**: Mid-sized.
- **Bison**: Capable, optimized for text and chat.
- **Unicorn**: Largest and most capable.

### Key Capabilities
- **Multilinguality**: Trained on text spanning over 100 languages.
- **Reasoning**: Improved logic, common sense reasoning, and mathematics capabilities.
- **Coding**: Pre-trained on a large quantity of source code, proficient in Python, JavaScript, and specialized languages like Verilog and Fortran.

## Comparison: PaLM 2 vs. Gemini
| Feature | PaLM 2 | Gemini |
| :--- | :--- | :--- |
| **Modalities** | Primarily Text (Unimodal) | Multimodal (Text, Image, Audio, Video) |
| **Architecture** | Transformer-based (Pathways) | Transformer-based (Native Multimodal) |
| **Reasoning** | Strong text-based logic | Enhanced multimodal reasoning |
| **Code Gen** | Proficient | Superior, integrated with more context |
| **Availability** | Legacy / Decommissioning | Active (Pro, Ultra, Nano, Flash) |

## Legacy & Transition
The PaLM API is decommissioned. Users are directed to migrate to Gemini, which offers superior performance, multimodal capabilities, and a more robust ecosystem.
