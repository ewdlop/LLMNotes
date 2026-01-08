# Research Papers and Resources

## Overview

This document provides a comprehensive collection of research papers, articles, and resources related to peripheral vision in AI, vision-language models, and efficient visual processing.

## Foundational Papers

### Attention Mechanisms

1. **"Attention Is All You Need"**
   - Authors: Vaswani et al.
   - Year: 2017
   - Venue: NeurIPS
   - Link: https://arxiv.org/abs/1706.03762
   - Key Contribution: Transformer architecture with self-attention
   - Relevance: Foundation for modern attention-based vision models

2. **"Recurrent Models of Visual Attention"**
   - Authors: Mnih, Heess, Graves, Kavukcuoglu
   - Year: 2014
   - Venue: NeurIPS
   - Link: https://arxiv.org/abs/1406.6247
   - Key Contribution: Reinforcement learning for selective visual attention
   - Relevance: Early work on computational foveation

3. **"Show, Attend and Tell"**
   - Authors: Xu et al.
   - Year: 2015
   - Venue: ICML
   - Link: https://arxiv.org/abs/1502.03044
   - Key Contribution: Attention mechanism for image captioning
   - Relevance: Spatial attention in vision-language tasks

### Vision Transformers

4. **"An Image is Worth 16x16 Words"** (Vision Transformer / ViT)
   - Authors: Dosovitskiy et al.
   - Year: 2020
   - Venue: ICLR 2021
   - Link: https://arxiv.org/abs/2010.11929
   - Key Contribution: Pure transformer for image classification
   - Relevance: Efficient visual token processing

5. **"Pyramid Vision Transformer"**
   - Authors: Wang et al.
   - Year: 2021
   - Venue: ICCV
   - Link: https://arxiv.org/abs/2102.12122
   - Key Contribution: Multi-scale feature pyramid for transformers
   - Relevance: Hierarchical visual processing

6. **"Swin Transformer"**
   - Authors: Liu et al.
   - Year: 2021
   - Venue: ICCV
   - Link: https://arxiv.org/abs/2103.14030
   - Key Contribution: Shifted windows for efficient attention
   - Relevance: Hierarchical vision transformer with local attention

### Efficient Vision Models

7. **"Perceiver: General Perception with Iterative Attention"**
   - Authors: Jaegle, Gimeno, Brock, et al.
   - Year: 2021
   - Venue: ICML
   - Link: https://arxiv.org/abs/2103.03206
   - Key Contribution: Cross-attention for efficient multimodal processing
   - Relevance: Information bottleneck mimics peripheral vision

8. **"Perceiver IO: A General Architecture for Structured Inputs & Outputs"**
   - Authors: Jaegle et al.
   - Year: 2021
   - Link: https://arxiv.org/abs/2107.14795
   - Key Contribution: Extended Perceiver to arbitrary outputs
   - Relevance: Efficient processing of high-dimensional inputs

9. **"EfficientNet: Rethinking Model Scaling"**
   - Authors: Tan & Le
   - Year: 2019
   - Venue: ICML
   - Link: https://arxiv.org/abs/1905.11946
   - Key Contribution: Compound scaling for efficient CNNs
   - Relevance: Efficiency in visual processing

### Vision-Language Models

10. **"CLIP: Learning Transferable Visual Models From Natural Language Supervision"**
    - Authors: Radford et al. (OpenAI)
    - Year: 2021
    - Venue: ICML
    - Link: https://arxiv.org/abs/2103.00020
    - Key Contribution: Contrastive vision-language pretraining
    - Relevance: Foundation for modern VLMs

11. **"Flamingo: A Visual Language Model for Few-Shot Learning"**
    - Authors: Alayrac et al. (DeepMind)
    - Year: 2022
    - Venue: NeurIPS
    - Link: https://arxiv.org/abs/2204.14198
    - Key Contribution: Perceiver Resampler for vision-language fusion
    - Relevance: Efficient multimodal fusion similar to peripheral vision

12. **"BLIP-2: Bootstrapping Language-Image Pre-training"**
    - Authors: Li et al. (Salesforce)
    - Year: 2023
    - Venue: ICML
    - Link: https://arxiv.org/abs/2301.12597
    - Key Contribution: Q-Former as information bottleneck
    - Relevance: Efficient vision-to-language bridging

13. **"LLaVA: Large Language and Vision Assistant"**
    - Authors: Liu et al.
    - Year: 2023
    - Link: https://arxiv.org/abs/2304.08485
    - Key Contribution: Simple yet effective VLM architecture
    - Relevance: Vision-language integration

14. **"PaLI: A Jointly-Scaled Multilingual Language-Image Model"**
    - Authors: Chen et al. (Google)
    - Year: 2022
    - Link: https://arxiv.org/abs/2209.06794
    - Key Contribution: Scaled vision-language model
    - Relevance: Large-scale multimodal training

### Foveation and Active Vision

15. **"Learning to Look Around"**
    - Authors: Jayaraman & Grauman
    - Year: 2018
    - Venue: NeurIPS
    - Link: https://arxiv.org/abs/1709.00507
    - Key Contribution: Active vision for embodied agents
    - Relevance: Sequential attention for scene understanding

16. **"Where to Look Next"**
    - Authors: Ramanishka et al.
    - Year: 2019
    - Venue: CVPR
    - Link: https://arxiv.org/abs/1904.04557
    - Key Contribution: Predicting next fixation points
    - Relevance: Foveation strategy learning

17. **"Foveated Neural Networks for Perception"**
    - Authors: Various
    - Year: Multiple papers 2017-2020
    - Key Contribution: Log-polar sampling, retina-inspired architectures
    - Relevance: Biologically-inspired foveation

18. **"Neural Module Networks for Reasoning over Text"**
    - Authors: Gupta & Lewis
    - Year: 2018
    - Key Contribution: Compositional reasoning with attention
    - Relevance: Hierarchical reasoning patterns

### Neuroscience-Inspired Vision

19. **"Deep Neural Networks Rival the Representation of Primate IT Cortex"**
    - Authors: Cadieu et al.
    - Year: 2014
    - Venue: PLOS Computational Biology
    - Link: https://doi.org/10.1371/journal.pcbi.1003963
    - Key Contribution: DNNs model biological vision
    - Relevance: Understanding peripheral vs foveal processing in brain

20. **"The Visual System's Eccentricity-Dependent Processing"**
    - Various neuroscience papers
    - Key Insight: Human vision degrades with eccentricity
    - Relevance: Biological foundation for peripheral vision models

### Object Detection and Segmentation

21. **"Faster R-CNN"**
    - Authors: Ren et al.
    - Year: 2015
    - Venue: NeurIPS
    - Link: https://arxiv.org/abs/1506.01497
    - Key Contribution: Region proposal networks
    - Relevance: Selective region processing

22. **"Segment Anything"** (SAM)
    - Authors: Kirillov et al. (Meta)
    - Year: 2023
    - Link: https://arxiv.org/abs/2304.02643
    - Key Contribution: Promptable segmentation
    - Relevance: Region selection for foveation

23. **"DETR: End-to-End Object Detection with Transformers"**
    - Authors: Carion et al.
    - Year: 2020
    - Venue: ECCV
    - Link: https://arxiv.org/abs/2005.12872
    - Key Contribution: Transformer-based detection
    - Relevance: Attention-based object localization

### Efficient Deep Learning

24. **"MobileNets"**
    - Authors: Howard et al.
    - Year: 2017
    - Link: https://arxiv.org/abs/1704.04861
    - Key Contribution: Depthwise separable convolutions
    - Relevance: Efficient mobile vision

25. **"SqueezeNet"**
    - Authors: Iandola et al.
    - Year: 2016
    - Link: https://arxiv.org/abs/1602.07360
    - Key Contribution: Compressed CNN architecture
    - Relevance: Model compression techniques

26. **"Network Pruning"**
    - Various papers on pruning and quantization
    - Relevance: Reducing computational requirements

### Video Understanding

27. **"SlowFast Networks"**
    - Authors: Feichtenhofer et al.
    - Year: 2019
    - Venue: ICCV
    - Link: https://arxiv.org/abs/1812.03982
    - Key Contribution: Dual-speed processing for video
    - Relevance: Multi-resolution temporal processing

28. **"TimeSformer"**
    - Authors: Bertasius et al.
    - Year: 2021
    - Venue: ICML
    - Link: https://arxiv.org/abs/2102.05095
    - Key Contribution: Space-time attention for video
    - Relevance: Efficient video transformers

## Surveys and Review Papers

29. **"Attention Mechanisms in Computer Vision"**
    - Authors: Guo et al.
    - Year: 2022
    - Venue: Computational Visual Media
    - Link: https://link.springer.com/article/10.1007/s41095-022-0271-y
    - Overview: Comprehensive survey of attention in CV

30. **"Vision-Language Pre-training: Current Trends and the Future"**
    - Authors: Du et al.
    - Year: 2022
    - Link: https://arxiv.org/abs/2202.09061
    - Overview: Survey of VLM architectures and training

31. **"Efficient Deep Learning"**
    - Various surveys on model compression, neural architecture search
    - Overview: Techniques for efficient inference

## Open-Source Projects and Code

### Frameworks and Libraries

1. **Hugging Face Transformers**
   - URL: https://github.com/huggingface/transformers
   - Models: BLIP, CLIP, LLaVA, and more
   - Documentation: https://huggingface.co/docs/transformers

2. **Perceiver PyTorch**
   - URL: https://github.com/lucidrains/perceiver-pytorch
   - Implementation of Perceiver and Perceiver IO

3. **Timm (PyTorch Image Models)**
   - URL: https://github.com/huggingface/pytorch-image-models
   - Vision transformers and efficient architectures

4. **OpenAI CLIP**
   - URL: https://github.com/openai/CLIP
   - Official CLIP implementation

5. **Meta SAM (Segment Anything)**
   - URL: https://github.com/facebookresearch/segment-anything
   - Segmentation for region proposal

### Example Implementations

6. **Attention Visualization Tools**
   - URL: https://github.com/jacobgil/pytorch-grad-cam
   - Visualize attention maps

7. **Visual Attention Mechanisms**
   - URL: https://github.com/MenghaoGuo/Awesome-Vision-Attentions
   - Collection of attention mechanisms

## Datasets and Benchmarks

### Vision-Language Datasets

1. **COCO (Common Objects in Context)**
   - URL: https://cocodataset.org/
   - Tasks: Detection, captioning, VQA
   - Scale: 330K images

2. **Visual Genome**
   - URL: https://visualgenome.org/
   - Content: Dense annotations, scene graphs
   - Scale: 108K images

3. **VQA (Visual Question Answering)**
   - URL: https://visualqa.org/
   - Task: Answer questions about images
   - Scale: 265K images

4. **RefCOCO / RefCOCO+**
   - Task: Referring expression comprehension
   - Relevance: Grounded language understanding

### Efficiency Benchmarks

5. **ImageNet**
   - URL: https://www.image-net.org/
   - Standard benchmark for vision models
   - Metrics: Accuracy, FLOPs, latency

6. **MMMU (Multimodal Multi-discipline Understanding)**
   - Task: College-level multimodal understanding
   - Relevance: Complex vision-language reasoning

## Online Resources

### Blogs and Tutorials

1. **Lil'Log - Attention Mechanisms**
   - URL: https://lilianweng.github.io/posts/2018-06-24-attention/
   - Comprehensive overview of attention

2. **distill.pub - Attention and Augmented RNNs**
   - URL: https://distill.pub/2016/augmented-rnns/
   - Interactive visualizations

3. **Papers With Code**
   - URL: https://paperswithcode.com/
   - Search: "visual attention", "efficient vision"
   - Leaderboards and implementations

### Video Lectures

4. **Stanford CS231n: Convolutional Neural Networks**
   - URL: http://cs231n.stanford.edu/
   - Coverage: Attention, vision architectures

5. **DeepMind Lectures on Deep Learning**
   - URL: https://www.youtube.com/deepmind
   - Topics: Perceiver, attention mechanisms

## Industry Applications

### Company Research

1. **Google Research - Vision**
   - Papers: PaLI, Perceiver, ViT
   - URL: https://research.google/research-areas/machine-perception/

2. **Meta AI - Vision**
   - Papers: SAM, DETR, SlowFast
   - URL: https://ai.facebook.com/research/

3. **DeepMind**
   - Papers: Flamingo, Perceiver, Gato
   - URL: https://www.deepmind.com/research

4. **OpenAI**
   - Papers: CLIP, GPT-4V
   - URL: https://openai.com/research

5. **Microsoft Research**
   - Papers: Florence, BEiT, Swin Transformer
   - URL: https://www.microsoft.com/en-us/research/

## Conferences and Workshops

### Major Venues

1. **CVPR** - Computer Vision and Pattern Recognition
2. **ICCV** - International Conference on Computer Vision
3. **ECCV** - European Conference on Computer Vision
4. **NeurIPS** - Neural Information Processing Systems
5. **ICML** - International Conference on Machine Learning
6. **ICLR** - International Conference on Learning Representations

### Relevant Workshops

- Vision and Language (VL)
- Efficient Deep Learning
- Neural Architecture Search
- Embodied AI
- Medical Imaging with Deep Learning

## Tools and Utilities

### Model Analysis

1. **FVCore (Facebook Vision Core)**
   - URL: https://github.com/facebookresearch/fvcore
   - Tools: FLOPs counting, parameter analysis

2. **TorchInfo**
   - URL: https://github.com/TylerYep/torchinfo
   - Model summary and statistics

3. **ONNX Runtime**
   - URL: https://onnxruntime.ai/
   - Cross-platform inference optimization

### Visualization

4. **Weights & Biases**
   - URL: https://wandb.ai/
   - Experiment tracking and visualization

5. **TensorBoard**
   - URL: https://www.tensorflow.org/tensorboard
   - Training visualization

## Books

1. **"Deep Learning for Vision Systems"**
   - Author: Mohamed Elgendy
   - Publisher: Manning
   - Year: 2020

2. **"Computer Vision: Algorithms and Applications"**
   - Author: Richard Szeliski
   - Publisher: Springer
   - Year: 2022

3. **"Deep Learning"**
   - Authors: Goodfellow, Bengio, Courville
   - Publisher: MIT Press
   - Year: 2016

## Future Directions and Recent Trends (2024-2025)

### Emerging Topics

1. **Multimodal Foundation Models**
   - GPT-4V, Gemini, Claude with vision
   - Unified architectures for any-to-any modalities

2. **Efficient Transformers**
   - Linear attention mechanisms
   - Sparse transformers
   - Hardware-aware designs

3. **Neural Architecture Search for Efficiency**
   - Automated design of efficient architectures
   - Task-specific optimization

4. **On-Device AI**
   - Mobile and edge deployment
   - Quantization and compression
   - Specialized hardware (NPUs, TPUs)

5. **Video-Language Models**
   - Temporal understanding
   - Efficient video processing
   - Action recognition and forecasting

## Community and Discussion

### Forums and Communities

1. **Reddit: r/MachineLearning, r/computervision**
2. **Hugging Face Forums**
3. **Papers With Code Community**
4. **AI Discord Servers**

### Twitter/X Accounts to Follow

- @ylecun (Yann LeCun)
- @karpathy (Andrej Karpathy)
- @jackrae (Jack Rae - DeepMind)
- @OriolVinyalsML (Oriol Vinyals)

## Contributing

To suggest additions to this resource list:
- Open an issue or PR in the repository
- Include: Title, Authors, Year, Link, Brief description
- Explain relevance to peripheral vision language models

---

**Last Updated**: January 2025
**Maintainer**: LLMNotes Project
**Version**: 1.0

This list is continuously updated as the field evolves.
