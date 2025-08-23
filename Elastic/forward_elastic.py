from pickle import EXT1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 07:08:50 2025

@author: dadd
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Set precision and random seeds
tf.keras.backend.set_floatx('float32')
np.random.seed(1234)
tf.random.set_seed(1234)

# Material properties (Titanium and SiC)
Eref=450e3#Mpa
Tref=5#N/mm
Lref=20#mm

E1 = 1
nu1=0.19

def lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu

#lam1, mu1 = lame_parameters(E1, nu1)
lam, mu = lame_parameters(E1, nu1)

class RJ_PINNs:
    def __init__(self, X, X_0, X_1, Y_0, Y_1, layers):
        # Domain points
        self.x = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
        self.y = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)

        # Boundary points
        self.xx_0 = tf.convert_to_tensor(X_0[:, 0:1], dtype=tf.float32)  # x=0
        self.yx_0 = tf.convert_to_tensor(X_0[:, 1:2], dtype=tf.float32)
        self.xx_1 = tf.convert_to_tensor(X_1[:, 0:1], dtype=tf.float32)  # x=L
        self.yx_1 = tf.convert_to_tensor(X_1[:, 1:2], dtype=tf.float32)
        self.xy_0 = tf.convert_to_tensor(Y_0[:, 0:1], dtype=tf.float32)  # y=0
        self.yy_0 = tf.convert_to_tensor(Y_0[:, 1:2], dtype=tf.float32)
        self.xy_1 = tf.convert_to_tensor(Y_1[:, 0:1], dtype=tf.float32)  # y=L
        self.yy_1 = tf.convert_to_tensor(Y_1[:, 1:2], dtype=tf.float32)

        # Network architecture
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Training tracking
        self.loss_hist = []
        self.iteration = 0

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers)-1):
            W = tf.Variable(tf.random.normal([layers[l], layers[l+1]],
                          dtype=tf.float32) * tf.sqrt(2./(layers[l]+layers[l+1])))
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X):
     H = X
     for l in range(len(self.layers)-2):
        W = self.weights[l]
        b = self.biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))

     W = self.weights[-1]
     b = self.biases[-1]
     Y = tf.add(tf.matmul(H, W), b)

     # Coordonnées normalisées [0,1]
     x_norm = X[:, 0:1]/1.0
     y_norm = X[:, 1:2]/1.0

     # Facteur pour condition Dirichlet sur bottom (y=0)
     dirichlet_factory = y_norm  # → 0 quand y=0
     dirichlet_factorx = x_norm  # → 0 quand x=0


     # Application des conditions
     u_x = Y[:, 0:1] * dirichlet_factorx  # u_x=0 seulement sur bottom
     u_y = Y[:, 1:2] * dirichlet_factory  # u_y=0 seulement sur bottom

     return tf.concat([
        u_x,                       # déplacement x (0 seulement sur bottom)
        u_y,                       # déplacement y (0 seulement sur bottom)

         ], axis=1)


    def model(self, x, y):
        X = tf.concat([x, y], axis=1)
        return self.neural_net(X)

    def get_strain(self, x, y):
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(y)
            u_pred = self.model(x, y)
            u_x = u_pred[:, 0:1]
            u_y = u_pred[:, 1:2]

        dux_dx = g.gradient(u_x, x)
        duy_dy = g.gradient(u_y, y)
        dux_dy = g.gradient(u_x, y)
        duy_dx = g.gradient(u_y, x)

        eps_xx = dux_dx
        eps_yy = duy_dy
        eps_xy = 0.5 * (dux_dy + duy_dx)
        #print(dux_dx)

        del g
        return eps_xx, eps_yy, eps_xy



    def R_physic(self, x, y):
     with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        g1.watch(y)
        eps_xx, eps_yy, eps_xy = self.get_strain(x, y)


        sig_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sig_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sig_xy = 2 * mu * eps_xy

     # Compute stress derivatives
     dsig_xx_dx = g1.gradient(sig_xx, x)
     dsig_xy_dy = g1.gradient(sig_xy, y)
     dsig_xy_dx = g1.gradient(sig_xy, x)
     dsig_yy_dy = g1.gradient(sig_yy, y)

     del g1

     # Verify gradients exist
     if None in [dsig_xx_dx, dsig_xy_dy, dsig_xy_dx, dsig_yy_dy]:
        raise ValueError("One or more stress derivatives could not be computed")

     eq1 = dsig_xx_dx + dsig_xy_dy
     eq2 = dsig_xy_dx + dsig_yy_dy

     return tf.concat([eq1, eq2], axis=0)


    def r_bc(self):


        # x=0 boundary (σ_xx = 0)
        eps_xx, eps_yy, eps_xy = self.get_strain(self.xx_0, self.yx_0)

        sig_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sig_xy = 2 * mu * eps_xy

        r_sig_xx=sig_xx-5
        r_sig_xy=sig_xy

        # x=L boundary (σ_xx = 0)
        eps_xx, eps_yy, eps_xy= self.get_strain(self.xx_1, self.yx_1)
        sig_xx = lam * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sig_xy = 2 * mu * eps_xy
        r_sig_xx1=sig_xx-1
        r_sig_xy1=sig_xy
        rb1=tf.concat([r_sig_xx1,r_sig_xy1],axis=0)

        # y=0 boundary (u = 0)
        u_pred = self.model(self.xy_0, self.yy_0)

        # y=L boundary (σ_yy = -5e6)
        eps_xx, eps_yy, eps_xy = self.get_strain(self.xy_1, self.yy_1)
        sig_yy = lam * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sig_yx=2*mu*eps_xy
        r_sig_yy=sig_yy-1
        r_sig_yx=sig_yx
        rb2=tf.concat([r_sig_yy,r_sig_yx],axis=0)


        return rb1,rb2

    def get_weights(self):
        variables = [*self.weights, *self.biases]
        return tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0)

    def set_weights(self, flat_weights):
        idx = 0
        for var in [*self.weights, *self.biases]:
            shape = tf.shape(var)
            size = tf.reduce_prod(shape)
            # Convert new_values to float32 before assigning
            new_values = tf.reshape(flat_weights[idx:idx + size], shape)
            # Use tf.cast to convert the tensor to float32
            new_values = tf.cast(new_values, dtype=tf.float32) # Changed this line
            var.assign(new_values)
            idx += size

    def R(self, p, print_loss=True):
        self.set_weights(p)

        r_physics = self.R_physic(self.x, self.y)
        rbc1,rbc2 = self.r_bc()

        r = tf.concat([r_physics, rbc1,rbc2], axis=0)
        loss = tf.reduce_mean(tf.square(r))

        if print_loss and self.iteration % 10 == 0:
            print(f"Iteration: {self.iteration}, Loss: {loss.numpy():.6e}")
            print("max r_physic",np.max(np.abs(r_physics.numpy())))
            print("max rbc1",np.max(np.abs(rbc1.numpy())))
            print("max rbc2",np.max(np.abs(rbc2.numpy())))
        self.iteration += 1
        return tf.reshape(r, [-1])

    def J(self, p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases])
            r = self.R(p, print_loss=False)

        jac_list = []
        for var in [*self.weights, *self.biases]:
            jac = tape.jacobian(r, var)
            if jac is not None:
                jac_flat = tf.reshape(jac, [r.shape[0], -1])
                jac_list.append(jac_flat)

        del tape
        return tf.concat(jac_list, axis=1).numpy()

    def train_trf(self, max_iter=500):

         p = self.get_weights().numpy()

         R0 = self.R(p).numpy()
         J0 = self.J(p)
         p_cp = - (J0.T @ R0) / (J0.T @ J0 @ (J0.T @ R0))  # Cauchy direction
         Theta_0 = np.linalg.norm(R0 + J0 @ p_cp)
         tr_options = {
        'eta1': 0.1,       # η₁ - relaxation pratique de α=10⁻⁴
        'eta2': 0.75,      # η₂ ≡ σ₂ de Celis
        'gamma1': 0.1,     # γ₁ ≡ τ₁ de Celis
        'gamma2': 2.0,     # γ₂ ≡ τ₃ de Celis
        'initial_trust_radius': 1.0,      # Δ₀
        'max_trust_radius': 100.0,        # Δ_max
        'initial_feasibility': Theta_0,        # Θ₀ (spécifique à Celis)
        #'gtol': 1e-8,      # ‖JᵀR‖ < ε (condition de convergence)
        #'xtol': 1e-8,      # ‖θₖ₊₁ - θₖ‖ < ε(1 + ‖θₖ₊₁‖)
        #'verbose': 2,
         }
         tr_options_rapide = {
           # STRATÉGIE AGGRESSIVE MAIS STABLE
    'eta1': 0.01,          # ↓ Plus strict (accepte seulement les bons pas)
    'eta2': 0.90,          # ↑ Plus agressif (expand plus facilement)
    'gamma1': 0.5,         # ↓ Moins de contraction (évite over-shrinking)
    'gamma2': 4.0,         # ↑ Expansion plus aggressive
    'initial_trust_radius': 0.1,  # ↓ Start petit mais croît vite
    'max_trust_radius': 1000.0,   # ↑ Permet une grande expansion
        }

         total_params=len(p)
         num_lambda_params=2
         lower_bounds = [-np.inf] * (total_params - num_lambda_params) + [ 0.0, 0.0]  # Example: lambda >= 0.1
         upper_bounds = [np.inf] * (total_params - num_lambda_params) + [ 5.0, 1.0]  # Example: lambda <= 10.0

         res = least_squares(
            fun=lambda p: self.R(p).numpy(),
            jac=lambda p: self.J(p),
            x0=p,
            method='trf',
            #tr_solver='exact',
            #max_nfev=max_iter,
            #verbose=2
            #tr_options=tr_options

            #bounds=(lower_bounds, upper_bounds)
         )
         self.set_weights(res.x)
         print(f" E estimated: {res.x[-2]*Eref}, v estimated:{res.x[-1]}")
         return res


    def predict(self, X_star):
        X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
        x_star = X_star_tf[:, 0:1]
        y_star = X_star_tf[:, 1:2]
        u_star = self.model(x_star, y_star)
        return u_star.numpy()

# Example usage
if __name__ == "__main__":
    # Create training data
    L = 1.0
    N_u = 500
    n_bc = 100
    data=np.load("elastic.npz")
    xy = data["coor"]
    #x = xy[:, 0].reshape(31,31)
    #y = xy[:, 1].reshape(31,31)
    x = np.unique( xy[:, 0]).reshape(-1, 1)
    y = np.unique( xy[:, 1]).reshape(-1, 1)

    X, Y = np.meshgrid(x, y)


    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]

    # Boundary points
    Y_0=np.hstack((X[0,:].flatten()[:, None], Y[0,:].flatten()[:, None]))
    #print('y0',Y_0.shape[0])
    n_ic=31
    n_bc=31
    #print('y piont',Y_0.shape)
    idy0 = np.random.choice(Y_0.shape[0], n_bc, replace=False)
    Y_train_0 = Y_0[idy0, :]
    #u_train = u_star[idx, :]


    Y_1=np.hstack((X[-1,:].flatten()[:, None], Y[-1,:].flatten()[:, None]))
    #print('y1',Y_1.shape[0])
    #n_ic=100

    idy1 = np.random.choice(Y_1.shape[0], n_bc, replace=False)
    Y_train_1 = Y_1[idy1, :]
    #u_train = u_star[idx, :]

    #n_bc=200
    #x=0
    X_0=np.hstack((X[:,0].flatten()[:, None], Y[:,0].flatten()[:, None]))
    #print('x sha',X_0.shape)
    idx0 = np.random.choice(X_0.shape[0], n_bc, replace=False)
    X_train_0 = X_0[idx0, :]


    #x=1
    X_1=np.hstack((X[:,-1].flatten()[:, None], Y[:,-1].flatten()[:, None]))

    idx1 = np.random.choice(X_1.shape[0], n_bc, replace=False)
    X_train_1 = X_1[idx1, :]

    # Create and train model
    layers = [2, 20, 20,2]  # Input: (x,y), Output: (u_x, u_y)
    model = RJ_PINNs(X_u_train, X_train_0, X_train_1, Y_train_0, Y_train_1, layers)
    model.train_trf(max_iter=500)

    # Predict and plot results
    u_pred = model.predict(X_star)
    u_x = u_pred[:, 0].reshape(X.shape)
    u_y = u_pred[:, 1].reshape(X.shape)
    np.savez("elasticp.npz",
         u_x=u_x,
         u_y=u_y,
         coor=xy
)


    Tref=5
    Eref=450e3
    Lref=20


    import matplotlib.pyplot as plt
    data_fem=np.load("elastic.npz")
    xy = data["coor"]
    x = np.unique( xy[:, 0]).reshape(-1, 1)
    y = np.unique( xy[:, 1]).reshape(-1, 1)

    X, Y = np.meshgrid(x, y)
    X*=Lref
    Y*=Lref

    uxf=data_fem["u_x"]*(Tref * Lref/ Eref)#.reshape(X_plot.shape)  # First output is u_x
    uyf=data_fem["u_y"]*(Tref * Lref/ Eref)#.reshape(X_plot.shape)  # First output is u_x


    data_pinn=np.load("elasticp.npz")
    ux=data_pinn["u_x"]*(Tref * Lref/ Eref)
    uy=data_pinn["u_y"]*(Tref * Lref/ Eref)
    uxf=uxf.reshape(X.shape)
    uyf=uyf.reshape(X.shape)


    ux=ux.reshape(X.shape)
    uy=uy.reshape(X.shape)

    print(uyf.shape)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, ux, levels=50, cmap='plasma')
    plt.colorbar(label='u_x (mm)')
    plt.title("Horizontal Displacement (u_x) RJ-PINNs")

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, uxf, levels=50, cmap='plasma')
    plt.colorbar(label='u_x (mm)')
    plt.title("Vertical Displacement (u_x) FEM")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, uy, levels=50, cmap='plasma')
    plt.colorbar(label='u_y (mm)')
    plt.title("Horizontal Displacement (u_y) RJ-PINNs")

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, uyf, levels=50, cmap='plasma')
    plt.colorbar(label='u_y (mm)')
    plt.title("Vertical Displacement (u_y) FEM")
    plt.show()



    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, uxf-ux, levels=50, cmap='plasma')
    plt.colorbar(label='u_x-uf (mm)')
    plt.title("Horizontal Displacement (u_x)")

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, uyf-uy, levels=50, cmap='plasma')
    plt.colorbar(label='u_y-uf (mm)')
    plt.title("Vertical Displacement (u_y)")
    plt.show()



    def compare_displacement(X, u_fem, uy, y_slice=20,title="u_y"):
     plt.figure(figsize=(8,5))
     plt.plot(X[y_slice,:], u_fem[y_slice,:], label='FEniCS', lw=2)
     plt.plot(X[y_slice,:], uy[y_slice,:], '--', label='RJ-PINNs', lw=2)
     plt.title(f"Comparison of {title} at y = {y_slice}")
     plt.xlabel("x (mm)")
     plt.ylabel(f"{title} (mm)")
     plt.grid()
     plt.legend()
     plt.show()

    compare_displacement(X, uxf, ux,10,'Horizontal Displacement (u_x)')
    compare_displacement(X, uyf,uy, 10,'Vertical Displacement (u_y)')

    #compare_displacement(X_plot, uxf, 10,'Vertical Displacement (u_y)')
    print("-----------rj-pinn--")

    print(f"Max u_x: {np.max(ux):.6f} mm")
    print(f"Max u_y: {np.max(uy):.6f} mm")
    print(f"Min u_x: {np.min(ux):.6f} mm")
    print(f"Min u_y: {np.min(uy):.6f} mm")

    ex = np.linalg.norm(uxf - ux, 2) / np.linalg.norm(uxf, 2)
    ey = np.linalg.norm(uyf - uy, 2) / np.linalg.norm(uyf, 2)

    print(ex)


    print(ey)

    print("-----------FEM---------------")

    print(f"Max u_x: {np.max(uxf):.6f} mm")
    print(f"Max u_y: {np.max(uyf):.6f} mm")
    print(f"Min u_x: {np.min(uxf):.6f} mm")
    print(f"Min u_y: {np.min(uyf):.6f} mm")










