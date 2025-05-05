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

## ⚠️ Important Consideration for RJ-PINNs Users

> 💡 **RJ-PINNs** improve convergence vs traditional PINNs — but **can still diverge** in some inverse problems. Read below 👇

<details>
<summary><strong>🔍 Why it happens & how to fix it</strong></summary>

- In **inverse problems** (e.g., identifying multiple parameters) or **complex direct problems**, RJ-PINNs can **diverge**.
- Cause: **Rapid decrease of the physics residual** `R_physics` leads to instability.

### 🛠️ Recommended Fixes
- 🔷 **Direct problems (no observed data):**
  - Lower `w_p` (e.g., `1e-1`, `1e-2`).

- 🔶 **Inverse problems:**
  - Increase `w_d` on `R_data` (e.g., `1e2`, `1e3`).
  - Lower `w_p` on `R_physics` (e.g., `1e-1`).

- 🧠 **Alternative:** Use adaptive weighting / normalization strategies like in traditional PINNs.

> 🧩 *This issue is common in PINNs — not specific to RJ-PINNs.*
</details>

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

