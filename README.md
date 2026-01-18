# Deep Learning From Scratch

A comprehensive collection of deep learning, machine learning, and AI implementations built from fundamental principles.

## Overview

This repository contains implementations and projects exploring neural networks, optimization algorithms, search methods, and natural language processing. Content spans multiple advanced computer science courses covering the theoretical foundations and practical applications.

## Coursework

### Machine Learning & Deep Learning
- **CSCI 566 (Deep Learning)** - Neural network architectures, backpropagation, gradient descent optimization
- **CSCI 567 (Machine Learning)** - Linear regression, neural networks from scratch, optimization comparisions
- **CSCI 445 (Intro to ML)** - Foundational ML algorithms and implementations

### Artificial Intelligence
- **CSCI 561 (Intro to AI)** - Search algorithms, reinforcement learning, constraint satisfaction, first-order logic
- **Stanford XCS221 (AI Principles)** - Advanced AI concepts and problem-solving techniques

## Key Projects

### Neural Network Implementations (A1-A4)
- Assignment implementations covering network architecture, training, and optimization
- Backpropagation from scratch with manual gradient computation
- Various optimization algorithms and their performance characteristics

### ML from Scratch (hw3)
- NeuralNetwork3.py and neuralnetworks.py implementations
- Custom optimizer implementations and comparisons
- Multiple datasets: circle, gaussian, spiral, XOR

### Search & AI Algorithms (hw1-hw3)
- TSP solver and pathfinding implementations
- Game-playing agents (Go, Tic-Tac-Toe) with Q-learning
- Constraint satisfaction and logic programming

### NLP & LLM Integration
- Twitter sentiment analysis pipeline (via submodule)
- LLM experimentation with Ollama and LangChain

## Technical Deep Dive

### Backpropagation Implementation
Core neural network implementations manually compute gradients using chain rule, providing visibility into mathematical foundations.

### Search Algorithms
Implementation of informed and uninformed search strategies with practical applications in pathfinding and optimization.

### Q-Learning
Reinforcement learning agents for game playing, exploring state-action value functions and policy optimization.

## Usage

```bash
# Run neural network implementations
python hw3/neuralnetworks.py

# Demo LLM with Ollama
cd twitter-sentiment-analysis
python demo.py

# Run Q-learning agents
cd hw2/tictactoe
python TicTacToe.py
```

## TODO

- [ ] Add comprehensive documentation for each assignment
- [ ] Implement additional optimization algorithms
- [ ] Add unit tests for core implementations
- [ ] Create visualization tools for network training
