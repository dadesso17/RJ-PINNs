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



##Schematic of RJ-PINNs



## RJ-PINNs Diagram

```mermaid
graph TD;
    A[Input: \( x_i,t \)] --> B[Hidden Layers];
    B --> C[Output: \( u_{\theta}(x_i,t) \)];
    
    C --> D1[Residual: \( \mathcal{R}_{\text{data}} \)];
    C --> D2[Residual: \( \mathcal{R}_{\text{physics}} \)];
    C --> D3[Residual: \( \mathcal{R}_{\text{bc}} \)];
    C --> D4[Residual: \( \mathcal{R}_{\text{ic}} \)];

    D1 --> E[Weighted Residuals: \( \mathbf{R} \)];
    D2 --> E;
    D3 --> E;
    D4 --> E;

    E --> F[Jacobian \( J = \frac{\partial \mathbf{R}}{\partial \theta} \)];
    F --> G[Optimization using TRF];







