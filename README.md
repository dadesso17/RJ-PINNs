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

Input: (x, t, u_obs(x_Ω, t), g(x_∂Ω, t), h(x_Ω, t₀))
      |
      v
+----------------+
| Hidden Layers  |
+----------------+
      |
      v
Output: u_θ(x, t)
      |
      |------> Residuals:
      |         ├── R_data:     u_θ - u_obs
      |         ├── R_physics:  F(u_θ, ∇u_θ, ...) - f
      |         ├── R_bc:       B(u_θ|∂Ω) - g
      |         └── R_ic:       I(u_θ|_{t₀}, ∂ₜu_θ|_{t₀}, ...) - h
      |
      v
Weighted Residual: R(θ)
      |
      v
Jacobian: J = ∂R / ∂θ
      |
      v
Optimization using TRF


## Citation
If you use RJ-PINNs in your research, please cite:

```bibtex
@software{Dadesso_RJ-PINNs_2025,
  author = {Dadesso, Dadoyi},
  title = {{Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs)}},
  year = {2025},
  version = {1.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15138086}
}
