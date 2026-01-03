# Google PaLM (Pathways Language Model)

Research and notes regarding Google's PaLM models.

> **Warning:** The PaLM API is decommissioned. The Vertex AI PaLM API is scheduled to be decommissioned in October 2024. Please upgrade to the Gemini API.

## Research Findings
See [Research_Notes.md](Research_Notes.md) for detailed research notes on architecture, training, and legacy.

## PaLM 2

PaLM 2 is a family of language models, optimized for ease of use on key developer use cases. It includes variations trained for text and chat generation as well as text embeddings.

### Model Sizes

PaLM 2 models come in different sizes, denoted by animal names:

*   **Bison**: The most capable PaLM 2 model size, suitable for text and chat tasks.
*   **Gecko**: The smallest, most efficient PaLM 2 model size, primarily used for embeddings.
*   **Otter**: A mid-sized model (often mentioned in broader contexts but Bison/Gecko are the main developer facing ones).
*   **Unicorn**: The largest model in the family.

### Model Variations

*   **Bison Text (`text-bison-001`)**:
    *   Optimized for language tasks such as code generation, text generation, text editing, problem-solving, recommendation generation, information extraction, and data extraction/generation.
    *   Input/Output: Text.
    *   Max input tokens: 8196.
    *   Max output tokens: 1024.
    *   Knowledge cutoff: mid-2021.

*   **Bison Chat (`chat-bison-001`)**:
    *   Optimized for dialog language tasks (chatbots, AI agents).
    *   Generates text in a conversational format.

*   **Gecko Embedding (`embedding-gecko-001`)**:
    *   Generates text embeddings for input text.
    *   Optimized for text up to 1024 tokens.

### Key Features

*   **Multilingual Capabilities**: Trained on multilingual text, enabling it to understand and generate idioms and nuances in over 100 languages.
    *   *Note: While the foundational model is multilingual, specific API implementations (like `text-bison-001`) may have had limited language support (e.g., English only) at launch or in certain regions.*
*   **Reasoning**: Capable of logic, common sense reasoning, and mathematics.
*   **Coding**: Proficient in coding tasks, trained on a large dataset of source code.

## Tools
*   `compare_palm_gemini.py`: A conceptual script demonstrating the comparison between PaLM 2 (Legacy) and Gemini (Active).

## References

*   [Google AI for Developers - PaLM 2 Models](https://ai.google.dev/palm_docs/palm)
*   [PaLM 2 Technical Report](https://ai.google/discover/palm2)
