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






**For inverse problems:**


*Instability can occur when attempting to identify multiple parameters simultaneously. It is easy to mitigate; the user can apply one of the techniques listed below or design a custom approach based on their knowledge.*



# RJ-PINNs Stability with Prior-Residuals

As described in the article, RJ-PINNs may incorporate regularization or a prior-based uncertainty formulation to enhance convergence when solving inverse problems



$$
\frac{\lambda_i - \mu_i}{\sigma_i}
$$

This acts as a **classical regularisation** in the deterministic sense,
and simultaneously corresponds to **Gaussian priors** in a Bayesian sense,
yielding a **MAP-type estimator** while keeping the standard ** RJ-PINN framework** intact.

## Mathematical Formulation

The augmented residual vector is:

$$
R_{\text{aug}}(\theta) = \begin{bmatrix}
R_{\text{data}} \\
R_{\text{physics}} \\
R_{\text{BC/IC}} \\
\frac{\lambda_1 - \mu_1}{\sigma_1} \\
\frac{\lambda_2 - \mu_2}{\sigma_2}
\end{bmatrix}
$$

The Least squares optimizer then minimizes:

$$
\min_{\theta} \| R_{\text{aug}}(\theta) \|_2^2
$$





```python
## Python Snippet

Imagine you need to identify **λ₁** and **λ₂** simultaneously, and the standard RJ-PINNs training shows instability.  
Suppose the true parameters are:

* `lambda1_true = 1.0`  
* `lambda2_true = 0.05`

### Parameter Initialization
As usual in RJ-PINNs, initialize the trainable parameters:

# Initialize trainable physical parameters
self.lambda1 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
self.lambda2 = tf.Variable(1.0, dtype=tf.float32, trainable=True)

# Set prior means and standard deviations
self.prior_mu1 = 1.0
self.prior_sigma1 = 0.1

self.prior_mu2 = 0.0
self.prior_sigma2 = 1.0

# Compute prior-residuals
rl1 = (self.lambda1 - self.prior_mu1) / self.prior_sigma1
rl2 = (self.lambda2 - self.prior_mu2) / self.prior_sigma2

rl1 = tf.reshape(rl1, [-1, 1])
rl2 = tf.reshape(rl2, [-1, 1])

# Combine all residuals into augmented residual vector
r = tf.concat([
    r_data,
 r_physic,
    r_bc,
    
    r_ic,
    rl1,
  rl2
], axis=0)



```



# Uniform prior on [a,b]




```python
## Python Snippet

Imagine you need to identify **λ₁** and **λ₂** simultaneously, and the standard RJ-PINNs training shows instability.  
Suppose the true parameters are:

* `lambda1_true = 1.0`  
* `lambda2_true = 0.05`

### Parameter Initialization
As usual in RJ-PINNs, initialize the trainable parameters:

# Initialize trainable physical parameters
self.lambda1 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
self.lambda2 = tf.Variable(1.0, dtype=tf.float32, trainable=True)

def uniform_prior_residual(param, low, high):
    # Résidu nul si dans l’intervalle, croissant si dehors
    penalty_low  = tf.nn.relu(low - param)
    penalty_high = tf.nn.relu(param - high)
    return penalty_low + penalty_high

# Exemple bornes [0,9]
rl1 = uniform_prior_residual(self.lambda1, 1.0, 9.0)
rl2 = uniform_prior_residual(self.lambda2, 0.0, 9.0)

rl1 = tf.reshape(rl1, [-1,1])
rl2 = tf.reshape(rl2, [-1,1])






# Combine all residuals into augmented residual vector
r = tf.concat([
    r_data,
 r_physic,
    r_bc,
    
    r_ic,
    rl1,
  rl2
], axis=0)



```


- **For direct problems:**
- RJ-PINNs can be sensitive to boundary conditions.

The real challenge in the RJ-PINNs framework is its sensitivity to boundary conditions. It often fails when confronted with certain real-world boundary conditions that are successfully handled by traditional numerical methods.
  

## Notice: The author of the RJ-PINNs framework declares that all publications currently on the RJ-PINNs project page are based on his own knowledge and research. If any content is found to be inappropriate or unsuitable for the page, he reserves the right to remove it and apologizes for any inconvenience or damage caused. He also welcomes contributions or collaborations that can help make RJ-PINNs more robust













  

  

## Citation
If you use RJ-PINNs in your research, please cite:

 Dadesso, Dadoyi, Residual Jacobian Physics-Informed Neural Networks (RJ-PINNs) for improved convergence and stability. Available at SSRN: https://ssrn.com/abstract=5506728 or http://dx.doi.org/10.2139/ssrn.5506728 
