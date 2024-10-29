# AISciVision: A Framework for Specializing Large Multimodal Models in Scientific Image Classification

AiSciVision is a general framework that enables Large Multimodal Models (LMMs) to adapt to niche image classification tasks.
The framework uses two key components: (1) Visual Retrieval-Augmented Generation (VisRAG) and (2) domain-specific tools utilized in an agentic workflow. 
To classify a target image, AiSciVision first retrieves the most similar positive and negative labeled images as context for the LMM. 
Then the LMM agent actively selects and applies tools to manipulate and inspect the target image over multiple rounds, refining its analysis before making a final prediction.

## Table of Contents
- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [API Keys](#api-keys)
  - [Experiments](#experiments)
- [Extending AiSciVision](#extending-aiscivision)
  - [Contributing](#contributing)
- [Codebase Overview](#codebase-overview)
  - [AiSciVision Implementation](#aiscivision-implementation)
  - [Baseline Implementations](#baseline-implementations)
  - [Utilities](#utilities)
- [License](#license)

## Installation

We recommend using Python 3.9+ and a CUDA-capable GPU.
Create a conda environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate aiscivision
```

## Running Experiments

### API Keys
The framework requires two API keys:
1. OpenAI API Key. Required for accessing GPT-4V or other OpenAI LMMs. Get your key at: https://platform.openai.com/api-keys
2. Google Maps API Key. Required for the satellite imagery tooling. To obtain:
    1. Create a Google Cloud Project
    2. Enable the Maps JavaScript API and Static Maps API
    3. Create credentials at: https://console.cloud.google.com/apis/credentials
    4. Enable billing (required for API access)

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY=`cat openai_api_key.txt`
export GMAPS_API_KEY=`cat gmaps_api_key.txt`
```

### Experiments

Run all baseline and AiSciVision experiments for a dataset.
The `solar` dataset is publicly available.
For `aquaculture` and `eelgrass` datasets, please contact the authors.
```bash
# replace <dataset> with: aquaculture, eelgrass, or solar
bash final_exps.sh <dataset>
```

## Extending AiSciVision
The framework is designed to be modular and extensible. 
Take these steps to **apply AiSciVision to your own dataset**:
1. Add dataset name and tools to `config.py`, and update parsing arguments in `utils.py`
2. Create dataset class in `dataloaders/datasets.py` implementing the abstract `ImageDataset` class
3. Define prompt schema in `promptSchema.py` inheriting from `BasePromptSchema`
4. Create tools in `tools/<dataset>.py` extending the `Tool` base class
5. Run experiments with `bash final_exps.sh <dataset>`

### Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the linting checks (`make lint` and `make fix-lint`)
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

The included `Makefile` provides utilities for maintaining code quality:
- `make lint`: Run code linting
- `make fix-lint`: Auto-fix linting issues
- `make find-todos`: Find TODO comments
- `make find-text SEARCH_STRING="aiscivision"`: Search codebase for specific text

## Codebase Overview

- `final_exps.py` and `final_exps.sh` execute all experiments.

### AiSciVision Implementation
- `main.py`: Experiment runner.
- `aiSciVision.py`: AiSciVision framework. Manages conversation state and history with LMM, and orchestrates between LMM, VisRAG system, and tool execution.
- `visualRAG.py`: Visual RAG system. Implements prompts for retrieval-augmented generation for visual tasks.
- `promptSchema.py`: Prompt Management. Defines prompt templates (visual context, tool use, initial/final prompts) for LMM use.
- `lmm.py`: Large Multimodal Model interface. Transforms conversation to LMM API parse-able turn-style conversation. Extensible to other APIs and models.
- `embeddingModel.py`: Embedding Models. Handles image preprocessing for the Visual RAG system.
- `tools/`: Tool definitions and implementations.

### Baseline Implementations
- `main_knn.py`. Experiment runner for KNN baseline. See model in `models/knn_classifier.py`.
- `main_clip_zero_shot.py`. Experiment runner for CLIP Zero Shot baseline. See model in `models/clip_classifier.py`.
- `main_clip_supervised.py`. Experiment runner for CLIP + MLP supervised model baseline. See model in `models/clip_classifier.py`.

### Utilities
- `config.py`. Common variables used throughout.
- `utils.py`. Experiment argument definitions, logging functions, evaluation metric functions.
- `create_test_set_selection.py`. Helper script to save an ordering of test samples, useful for reproducing experiments.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
