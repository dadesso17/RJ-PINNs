# RJ-PINNs
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



