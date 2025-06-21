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
  - 🔧 Decrease the weight `w_p` applied to `R_physics` (e.g., `1e-3`).
  - - **For direct problems with observed data:**
  - 🔼 Increase the weight `w_d` on `R_data` (e.g., `1e4`,).
  - 🔽 Decrease the weight `w_p` on `R_physics` (e.g., set to `1e-1...`).



- **For inverse problems:**
  - 🔼 Increase the weight `w_d` on `R_data` (e.g., `1e4`,).
  - 🔽 Decrease the weight `w_p` on `R_physics` (e.g., set to `1e-1`).

- 🧠 **Alternative strategies:**
  - Use **adaptive weighting techniques** or **normalization strategies** (as in traditional PINNs) to improve stability.

> 🧩 **Note:** This issue is common in the general PINN framework — it's **not specific to RJ-PINNs**.

# Lemma: Stability of Adaptive Weights in Physics-Informed Neural Networks

## Formal Statement

Let:
- $R_i^{(k)}$ be the residual of type $i$ at training step $k$
- $\|R_i^{(k)}\|_\infty = \max|R_i^{(k)}(x,t)|$ be the infinity norm

Define the **logarithmic magnitude**:
$$
\alpha_i^{(k)} = \log_{10}\left(\|R_i^{(k)}\|_\infty + \epsilon\right)
\quad (\epsilon=10^{-16}\text{ for numerical stability})
$$

The **adaptive weights** are computed as:
$$
w_i^{(k)} = \text{clip}\left(
10^{\eta(\alpha_i^{(k)} - \alpha_{\text{target}})},\ 
w_{\min},\ 
w_{\max}
\right)
$$

**Where**:
- $\eta \in (0,1]$: Adaptation rate (typically 0.1)
- $\alpha_{\text{target}}$: Target order (e.g., -3 for $10^{-3}$)
- $[w_{\min}, w_{\max}]$: Weight bounds (e.g., $[10^{-6}, 10^6]$)

**Then**:
$$
\limsup_{k \to \infty} \left|w_i^{(k)} R_i^{(k)}\right| \leq 10^{\alpha_{\text{target}}}
\quad \forall i
$$

---

## Physical Interpretation

| Property            | Description                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **Balance**         | Prevents any single residual from dominating the loss function              |
| **Automatic Scaling** | Dynamically adjusts weights to maintain target scale                      |
| **Numerical Stability** | Clipping avoids extreme weight values that could destabilize training    |

---

## Proof Sketch

1. **Boundedness**:
   Since $\|R_i^{(k)}\|_\infty \leq C$ (by problem physics), then:
   $$
   \alpha_i^{(k)} \leq \log_{10}(C + \epsilon)
   $$

2. **Weight Behavior**:
   - When $\alpha_i^{(k)} \gg \alpha_{\text{target}}$:
     $$ w_i^{(k)} \nearrow w_{\max} \text{ exponentially} $$
   - When $\alpha_i^{(k)} \ll \alpha_{\text{target}}$:
     $$ w_i^{(k)} \searrow w_{\min} \text{ exponentially} $$

3. **Product Control**:
   The weighted residual satisfies:
   $$
   |w_i^{(k)} R_i^{(k)}| \approx 10^{\eta(\alpha_i^{(k)}-\alpha_{\text{target}})} \cdot 10^{\alpha_i^{(k)}}} = 10^{(1+\eta)\alpha_i^{(k)} - \eta\alpha_{\text{target}}}}
   $$
   which remains bounded near $10^{\alpha_{\text{target}}}$ when $\eta \leq 1$.

---

## Implementation in PINNs

```python
def calculate_weights(current_residual):
    current_order = np.log10(np.max(np.abs(current_residual)) + 1e-16
    weight = 10 ** (eta * (current_order - target_order))
    return np.clip(weight, min_weight, max_weight)




## Citation
If you use RJ-PINNs in your research, please cite:

```bibtex
@software{Dadesso_RJ-PINNs_2025,
  author = {Dadesso, D.},
  year = {(2025)},

  title = {{Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs) for Guaranteed Convergence}},
  version = {1.0},
  publisher = {Zenodo},
  doi = { 10.5281/zenodo.15138086},

}

