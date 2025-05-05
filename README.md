# RJ-PINNs: Residual Jacobian Physics-Informed Neural Networks for Guaranteed Convergence

[![Stars](https://img.shields.io/github/stars/dadesso17/RJ-PINNs?style=social)](https://github.com/dadesso17/RJ-PINNs/stargazers)
[![Forks](https://img.shields.io/github/forks/dadesso17/RJ-PINNs?style=social)](https://github.com/dadesso17/RJ-PINNs/network/members)
[![Issues](https://img.shields.io/github/issues/dadesso17/RJ-PINNs)](https://github.com/dadesso17/RJ-PINNs/issues)
[![License](https://img.shields.io/github/license/dadesso17/RJ-PINNs)](https://github.com/dadesso17/RJ-PINNs/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/dadesso17/RJ-PINNs)](https://github.com/dadesso17/RJ-PINNs/commits/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15138086.svg)](https://doi.org/10.5281/zenodo.15138086)

RJ-PINNs, introduced by Dadoyi Dadesso, is a pioneering framework using Jacobian-based least-squares (TRF) to directly minimize residuals without traditional loss functions. This method is the first to eliminate gradient optimizers in PINNs, offering unmatched robustness for inverse PDE problems.

For more information, please refer to the following:https://github.com/dadesso17/RJ-PINNs/ or preprint: https://doi.org/10.5281/zenodo.15138086

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
| **Convergence** | Not guaranteed | Robust convergence |

*Table: Comparison between Traditional PINNs and RJ-PINNs*

  ## RJ-PINNs Diagram  
    <p align="center">
  <img src="./im.png" width="800">
</p>

## ⚠️ Important: Practical Notes for Using RJ-PINNs

> **RJ-PINNs** (Residual Jacobian Physics-Informed Neural Networks) offer better convergence and stability than traditional PINNs — but some practical issues can still arise.

### 🚨 When Problems Occur
- In **inverse problems** involving **multiple parameter identification**, or in some **complex direct problems**, **RJ-PINNs may still diverge**.
- This is often caused by a **rapid decrease of the physics residual** `R_physics`, leading to **instability or divergence**.

### ✅ How to Fix It
- **For direct problems without observed data:**
  - 🔧 Decrease the weight `w_p` applied to `R_physics` (e.g., `1e-1`, `1e-2`, ...).

- **For inverse problems:**
  - 🔼 Increase the weight `w_d` on `R_data` (e.g., `1e2`, `1e3`, ...).
  - 🔽 Decrease the weight `w_p` on `R_physics` (e.g., set to `1e-1`).

- 🧠 **Alternative strategies:**
  - Use **adaptive weighting techniques** or **normalization strategies** (as in traditional PINNs) to improve stability.

> 🧩 **Note:** This issue is common in the general PINN framework — it's **not specific to RJ-PINNs**.


## Citation
If you use RJ-PINNs in your research, please cite:

```bibtex
@software{Dadesso_RJ-PINNs_2025,
  author = {Dadesso, Dadoyi},
  title = {{Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs) for Guaranteed Convergence}},
  version = {1.0},
  publisher = {Zenodo},
  doi = { 10.5281/zenodo.15138086},
  year = {2025}

}

