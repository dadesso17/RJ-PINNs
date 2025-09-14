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
- RJ-PINNs can be sensitive to boundary conditions.
  


**For inverse problems:**


# RJ-PINNs Stability with Prior-Residuals

To enhance stability we append **prior-residuals** to the RJ-PINNs residual vector:

$$
\frac{\lambda_i - \mu_i}{\sigma_i}
$$

This acts as a **classical regularisation** in the deterministic sense,
and simultaneously corresponds to **Gaussian priors** in a Bayesian sense,
yielding a **MAP-type estimator** while keeping the standard **Gauss–Newton RJ-PINN framework** intact.

## Mathematical Formulation

The augmented residual vector is:

$$
R_{\text{aug}}(\theta) = \begin{bmatrix}
R_{\text{data}} \\
R_{\text{physics}} \\
R_{\text{BC/IC}} \\
\frac{\lambda_1 - \mu_1}{\sigma_1} \\
\frac{\lambda_3 - \mu_2}{\sigma_2}
\end{bmatrix}
$$

The Gauss–Newton solver then minimizes:

$$
\min_{\theta} \| R_{\text{aug}}(\theta) \|_2^2
$$

## Python Snippet

```python
# Assume tf is TensorFlow, r_data etc. are existing residual tensors
rl1 = (lambda_3 - prior_mu2) / prior_sigma2
rl2 = (lambda_1 - prior_mu1) / prior_sigma1

rl1 = tf.reshape(rl1, [-1, 1])
rl2 = tf.reshape(rl2, [-1, 1])

# Combine residuals
r = tf.concat([r_data, r_physic, r_bc0, r_bc0_xx,
               r_bc1, r_bc1_xx, r_ic, r_ic_t,
               rl1, rl2], axis=0)
```

Place this content in your project **README.md** so it appears on the repository home page and on your GitHub Pages site if Pages is configured to use the root `README.md` as index.














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
