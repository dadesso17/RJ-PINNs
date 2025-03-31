## RJ-PINNs
I introduce RJ-PINNs: A breakthrough PINNs framework using Jacobian-based least-squares (TRF) to directly minimize residuals without traditional loss functions. First method to eliminate gradient optimizers in PINNs, offering unmatched robustness for inverse,direct PDE problems 


For more information, please refer to the following:(https://github.com/dadesso17/RJ-PINNs/)

## Key Differences

| Aspect         | Traditional PINNs                         | RJ-PINNs                               |
|--------------|--------------------------------|--------------------------------|
| **Objective** | Minimize a loss function L(θ) | Minimize the residuals R(θ) |
| **Gradient** | Compute ∇L(θ) | Compute ∇R(θ) |
| **Optimization** | Use gradient-based optimizers (e.g., Adam, L-BFGS) | Use least-squares optimizer (e.g., TRF) |
| **Implementation** | Define a loss function and its gradient | Define residuals and their Jacobian |
| **Convergence** | Not guaranteed | Robust convergence |

*Table: Comparison between Traditional PINNs and RJ-PINNs*



# RJ-PINNs: Residual-Jacobian Physics-Informed Neural Networks

A framework for solving differential equations using Physics-Informed Neural Networks (PINNs) with a focus on residual and Jacobian computation for optimization.

## Overview

This repository implements RJ-PINNs, which combine neural networks with physics-based constraints to solve differential equations. The approach computes various residuals and uses their Jacobian for optimization with Trust Region Reflective (TRF) methods.

## Architecture

```mermaid
graph TD
    A[Input: xᵢ, t] --> B[Hidden Layers]
    B --> C[Output: uθ(xᵢ,t)]
    C --> D[Residual: R_data]
    C --> E[Residual: R_physics]
    C --> F[Residual: R_bc]
    C --> G[Residual: R_ic]
    D --> H[Weighted Residuals: R]
    E --> H
    F --> H
    G --> H
    H --> I[Jacobian J = ∂R/∂θ]
    I --> J[Optimization using TRF]
