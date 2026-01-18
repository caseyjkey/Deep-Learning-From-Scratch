# Deep Learning From Scratch

A comprehensive collection of deep learning implementations from USC CSCI 566 and Stanford XCS221, featuring neural network architectures built from fundamental principles.

## Overview

This repository contains coursework and projects implementing deep learning concepts from scratch, including backpropagation, gradient descent, and modern NLP techniques with LLM experimentation.

## Contents

- **A1-A4**: Neural network assignments implementing core architectures
- **assignments**: Additional coursework and implementations
- **nlp.py**: Natural language processing with sentiment analysis
- **demo.py**: LLM experimentation using Ollama
- **flock.py**: Distributed training implementations

## Key Features

- Neural network architectures implemented from scratch
- Backpropagation and gradient descent implementations
- NLP sentiment analysis pipeline
- LLM integration with Ollama and LangChain

## Technical Deep Dive

### Backpropagation Implementation
The core neural network implementations manually compute gradients using chain rule, providing visibility into the mathematical foundations of deep learning.

### NLP Pipeline
The sentiment analysis system includes:
- Tokenization and preprocessing
- Ensemble model approaches
- Twitter streaming integration

### LLM Experimentation
Uses Ollama for local LLM inference with LangChain integration for prototyping AI applications.

## Usage

```bash
# Run NLP sentiment analysis
python nlp.py

# Demo LLM with Ollama
python demo.py
```

## TODO

- [ ] Migrate implementations to PyTorch
- [ ] Add Transformer architecture implementations
- [ ] Implement attention mechanisms from scratch
- [ ] Add comprehensive unit tests
- [ ] Document architecture decisions
