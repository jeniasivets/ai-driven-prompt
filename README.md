# AI-driven Prompting Strategies for Image Generation

A comprehensive framework for image synthesis, prompt optimization, and evaluation, leveraging the SDXL model.

## Overview

This project implements an automated pipeline for:
- Prompt refinement using multiple strategies
- Quality assessment through diverse metrics
- Benchmark testing across varying prompt difficulty levels



## Project Structure

```plaintext
├── src/
│   ├── prompt_refiner.py      # Prompt enhancement strategies
│   ├── evaluator.py           # Evaluation pipeline
│   ├── run_generate.py        # Generate benchmark execution
│   └── run_evaluate.py        # Evaluate benchmark execution
├── data/
│   └── benchmark_prompts.json # Prompts categorized by difficulty
├── results/                   # Stores generated images
└── notebooks/                 # Visualization
```

## Model Selection

Utilizes **Stable Diffusion XL (SDXL)**, chosen for its strong community support, ease of deployment, and balanced performance, making it suitable for local experimentation.

## Features

### 1. Prompt Enhancement
- **LLM-based Prompt Refinement**: Tailors user prompts to model capabilities
- **Attention Map Manipulation**: Implements Prompt-to-Prompt techniques for enhanced control [[Link]](https://prompt-to-prompt.github.io)
  - Techniques include reweighting attention for specific tokens
- **Batch Generation**: Multiple seeds for optimal results

### 2. Image Generation
- Integration with diffusers `StableDiffusionXLPipeline`
- Support for various generation parameters
- Attention control mechanisms
- Automated pipeline execution

### 3. Evaluation Pipeline
- CLIP score for prompt-image alignment [[Link]](https://github.com/linzhiqiu/t2v_metrics)
- TIFA score for detailed scene understanding [[Link]](https://tifa-benchmark.github.io).
- Inference time tracking
- Analyzes performance based on prompt difficulty (see *Benchmark Design*).

### 4. Benchmark Design

Prompts are categorized by difficulty and generated automatically:

- **Low**: Simple objects and scenes, minimal lighting.
- **Medium**: Moderate complexity, multiple objects, facial expressions.
- **High**: Complex multi-object scenes, detailed environments, advanced lighting.

