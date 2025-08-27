# RJ-PINNs: Residual Jacobian Physics-Informed Neural Networks 


RJ-PINNs, introduced by Dadoyi Dadesso, is a pioneering framework using Jacobian-based least-squares (TRF) to directly minimize residuals without traditional loss functions. This method is the first to eliminate gradient optimizers in PINNs, offering unmatched robustness for inverse PDE problems.

For more information, please refer to the following:https://github.com/dadesso17/RJ-PINNs/ or Dadesso, D. (2025).https://doi.org/10.5281/zenodo.16956061

## Features
- **Jacobian-based least-squares (TRF)**: Directly minimize residuals without traditional loss functions.
- **No Gradient Optimizers**: Eliminate the need for gradient optimizers in PINNs.
- **Robustness for Inverse PDE Problems**: Unmatched robustness for solving inverse partial differential equations (PDE) problems.

## Key Differences

| Aspect         | Traditional PINNs                         | RJ-PINNs                               |
|--------------|--------------------------------|--------------------------------|
| **Objective** | Minimize a loss function L(θ) | Minimize the residuals R(θ) |
| **Gradient** | Compute ∇L(θ) | Compute ∇R(θ) |
| **Optimization** | Use gradient-based optimizers (e.g., Adam, L-BFGS) | Use least-squares optimizer (e.g., TRF) |
| **Implementation** | Define a loss function and its gradient | Define residuals and their Jacobian |
| **Convergence** | Not guaranteed | Improved|

*Table: Comparison between Traditional PINNs and RJ-PINNs*

  ## RJ-PINNs Diagram  
    <p align="center">
  <img src="./im.png" width="800">
</p>
#### ⚠️ Important: Practical Notes for Using RJ-PINNs

> **RJ-PINNs** (Residual Jacobian Physics-Informed Neural Networks) offer better convergence and stability than traditional PINNs — but some practical issues can still arise.


- **For direct problems:**
- It is recommended to apply a non-dimensionalization technique, especially when the physical parameters have large magnitudes.
- RJ-PINNs can be sensitive to boundary conditions.
  


**For inverse problems:**
*Instability can occur when attempting to identify multiple parameters simultaneously. Possible mitigation strategies include*
- 
  - 🔼 Increase the weight `w_d` on `R_data` (e.g., `1e1`....).
  - 🔽 Decrease the weight `w_p` on `R_physics` (e.g., set to `1e-1...).
  - 
    *These adjustments, however, are problem-dependent*
    
*NB: RJ-PINNs are generally reliable when identifying a single parameter. However, the simultaneous identification of multiple parameters may require manual tuning. The convergence efficiency of RJ-PINNs in inverse problems remains an open question*

## 🧠  Stability of Adaptive Weights in Residual Jacobian Physics-Informed Neural Networks
  

  

## Citation
If you use RJ-PINNs in your research, please cite:

Dadesso, D. (2025). Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs) for improved convergence and stability. Zenodo. https://doi.org/10.5281/zenodo.16956061
