# 🧠 MLP Presentation - Interactive Neural Network Visualizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Watch backpropagation come alive! Step through every calculation of a neural network in real-time.</b>
</p>

<div align="center">
  <img src="network_diagram.png" alt="MLP Architecture" width="600"/>
  <p><i>Interactive 2-2-1 Neural Network Architecture with Live Updates</i></p>
</div>

## 🎯 What is This?

An interactive educational tool that **visualizes every single calculation** inside a neural network during training. Perfect for students, teachers, and anyone who wants to truly understand how backpropagation works.

## ✨ Key Features

| Feature | What it Shows |
|---------|--------------|
| **Live Network Graph** | Real-time weights, activations, and connections |
| **Step-by-Step Math** | Every equation with current numbers plugged in |
| **Manual Training** | Control each iteration - forward and backward pass |
| **Weight History** | Track how weights evolve during training |
| **Interactive Inputs** | Change x₁, x₂, target y, and learning rate on the fly |

## 🧮 The Neural Network Mathematics

### Network Architecture

<div align="center">
  <img src="mlp_equations.png" alt="Network Equations" width="700"/>
</div>

### Forward Pass Equations

<table>
<tr>
<th>Layer</th>
<th>Computation</th>
<th>Activation</th>
</tr>
<tr>
<td>Hidden 1</td>
<td>

$z_{h_1} = x_1 w_{00} + x_2 w_{01} + b_1$

</td>
<td>

$h_1 = \sigma(z_{h_1}) = \frac{1}{1+e^{-z_{h_1}}}$

</td>
</tr>
<tr>
<td>Hidden 2</td>
<td>

$z_{h_2} = x_1 w_{10} + x_2 w_{11} + b_1$

</td>
<td>

$h_2 = \sigma(z_{h_2})$

</td>
</tr>
<tr>
<td>Output</td>
<td>

$z_o = h_1 v_0 + h_2 v_1 + b_2$

</td>
<td>

$\hat{y} = \sigma(z_o)$

</td>
</tr>
<tr>
<td colspan="3" align="center">

**Loss Function:** $L = (y - \hat{y})^2$

</td>
</tr>
</table>

### Backpropagation Chain Rule

<div align="center">

$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_o} \cdot \frac{\partial z_o}{\partial h} \cdot \frac{\partial h}{\partial z_h} \cdot \frac{\partial z_h}{\partial w}$

</div>

### Complete Backpropagation Flow

```mermaid
graph TD
    A[Loss L = (y - ŷ)²] --> B[dL/dŷ = -2(y-ŷ)]
    B --> C[δₒ = dL/dŷ · σ'(zₒ)]
    C --> D[dL/dv = δₒ · h]
    C --> E[δₕ = δₒ · v · σ'(zₕ)]
    E --> F[dL/dw = δₕ · x]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#fbb,stroke:#333
    style E fill:#fbb,stroke:#333
    style F fill:#fbb,stroke:#333
Gradient Descent Update
<div align="center">
$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$

$v_{new} = v_{old} - \eta \cdot \frac{\partial L}{\partial v}$

</div>
