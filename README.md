# MLP Presentation - Interactive Neural Network Visualizer

An interactive Streamlit application that visualizes the inner workings of a Multilayer Perceptron (MLP) with one hidden layer. Watch forward propagation, backpropagation, and gradient descent in action with step-by-step calculations and real-time network visualization.

![MLP Presentation Screenshot](screenshot.png)

## Features

- **Interactive Network Visualization**: Real-time display of the neural network architecture with weighted connections and neuron activations
- **Step-by-Step Computation**: Detailed breakdown of:
  - Forward propagation calculations
  - Loss computation
  - Backpropagation through output and hidden layers
  - Gradient descent updates
- **Manual Training Control**: Step through iterations one at a time or reset to random weights
- **Live Parameter Display**: Current values for inputs, weights, biases, and activations
- **Mathematical Notation**: Clear LaTeX equations explaining each step of the process
- **Weight History**: Track weight changes across iterations

## How It Works

The application implements a simple neural network with:
- 2 input neurons (x₁, x₂)
- 2 hidden neurons (h₁, h₂) with sigmoid activation
- 1 output neuron (ŷ) with sigmoid activation
- Mean squared error loss function
- Gradient descent optimization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mlp-presentation.git
cd mlp-presentation
