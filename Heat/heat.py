

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
from scipy.optimize import least_squares, root, minimize

# Set precision to float32
tf.keras.backend.set_floatx('float32')

seeds_num=1234
np.random.seed(seeds_num)
tf.random.set_seed(seeds_num)

class RJ_PINNs:
    def __init__(self, X, u,X_bc0,X_bc1,X_ic,layers,init_method='xavier',  lambda_init_method='constant'):

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u

        # Define placeholders for inputs (compatible with TensorFlow v2)
        self.x = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.t = tf.convert_to_tensor(self.t, dtype=tf.float32)
        self.u= tf.convert_to_tensor(self.u, dtype=tf.float32)



        # Define placeholders for inputs (compatible with TensorFlow v2)
        self.x_bc0 = tf.convert_to_tensor(X_bc0[:, 0:1], dtype=tf.float32)
        self.t_bc0 = tf.convert_to_tensor(X_bc0[:, 1:2], dtype=tf.float32)




        self.x_bc1 = tf.convert_to_tensor(X_bc1[:, 0:1], dtype=tf.float32)
        self.t_bc1 = tf.convert_to_tensor(X_bc1[:, 1:2], dtype=tf.float32)





        self.x_ic = tf.convert_to_tensor(X_ic[:, 0:1], dtype=tf.float32)
        self.t_ic = tf.convert_to_tensor(X_ic[:, 1:2], dtype=tf.float32)

        self.layers = layers

        # Initialize neural network
        self.weights, self.biases = self.initialize_NN(layers,init_method='xavier')

        # Variables for optimization
        if lambda_init_method == 'constant':
            self.lambda_1 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
            self.lambda_2 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
            self.lambda_3 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        elif lambda_init_method == 'uniform':
            self.lambda_1 = tf.Variable(tf.random.uniform([1], minval=0.0, maxval=2.0), dtype=tf.float32, trainable=True)
            self.lambda_2 = tf.Variable(tf.random.uniform([1], minval=0.0, maxval=2.0), dtype=tf.float32, trainable=True)
            self.lambda_3 = tf.Variable(tf.random.uniform([1], minval=0.0, maxval=2.0), dtype=tf.float32, trainable=True)
        elif lambda_init_method == 'normal':
            self.lambda_1 = tf.Variable(tf.random.normal([1], mean=1.0, stddev=0.1), dtype=tf.float32, trainable=True)
            self.lambda_2 = tf.Variable(tf.random.normal([1], mean=1.0, stddev=0.1), dtype=tf.float32, trainable=True)
            self.lambda_3 = tf.Variable(tf.random.normal([1], mean=1.0, stddev=0.1), dtype=tf.float32, trainable=True)
        elif lambda_init_method == 'physical':
            self.lambda_1 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
            self.lambda_2 = tf.Variable(1.0, dtype=tf.float32, trainable=True)
            self.lambda_3 = tf.Variable(0.5, dtype=tf.float32, trainable=True)
        else:
            raise ValueError("Méthode d'initialisation des paramètres externes non reconnue.")



        self.param_hist1 = []
        self.param_hist2 = []
        self.loss_hist = []
        self.iteration_hist = []
        self.err_l1 = []
        self.err_l2 = []
        self.iteration = 0

    def initialize_NN(self, layers, init_method):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            if init_method == 'xavier':
                W = self.xavier_init(size=[layers[l], layers[l + 1]])
            elif init_method == 'he':
                W = self.he_init(size=[layers[l], layers[l + 1]])
            elif init_method == 'uniform':
                W = self.uniform_init(size=[layers[l], layers[l + 1]])
            elif init_method == 'normal':
                W = self.normal_init(size=[layers[l], layers[l + 1]])
            elif init_method == 'orthogonal':
                W = self.orthogonal_init(size=[layers[l], layers[l + 1]])
            else:
                raise ValueError("Méthode d'initialisation non reconnue.")

            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev))

    def he_init(self, size):
        in_dim = size[0]
        he_stddev = np.sqrt(2 / in_dim)
        return tf.Variable(tf.random.truncated_normal([in_dim, size[1]], stddev=he_stddev))

    def uniform_init(self, size):
        in_dim = size[0]
        limit = np.sqrt(1 / in_dim)
        return tf.Variable(tf.random.uniform([in_dim, size[1]], minval=-limit, maxval=limit))

    def normal_init(self, size):
        return tf.Variable(tf.random.normal(size, mean=0.0, stddev=0.01))

    def orthogonal_init(self, size):
        W = tf.random.normal(size)
        W, _, _ = tf.linalg.svd(W)  # Décomposition SVD pour orthogonaliser
        return tf.Variable(W)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def R_data(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u
    

    def R_physic(self, x, t):
        lambda_3 = self.lambda_3
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = self.R_data(x, t)
            u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
            #u_xxx = tape.gradient(u_xx, x)
        #u_tt = tape.gradient(u_t, t)

        #u_xxxx = tape.gradient(u_xxx, x)
        del tape #deleted the tape after it is used

        f = u_t - lambda_3**2 * u_xx

        return f
    
    def R_bc(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u

    def R_ic(self, x, t):
        X = tf.concat([x, t], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u
    def get_weights(self):
        theta = [*self.weights, *self.biases, self.lambda_3]
        return tf.concat([tf.reshape(var, [-1]) for var in theta], axis=0)

    def set_weights(self, flat_weights):
        idx = 0
        for var in [*self.weights, *self.biases, self.lambda_3]:
            shape = tf.shape(var)
            size = tf.reduce_prod(shape)
            new_values = tf.reshape(flat_weights[idx:idx + size], shape)
            new_values = tf.cast(new_values, dtype=tf.float32)
            var.assign(new_values)
            idx += size

    def R(self, p, print_loss=True):
     self.set_weights(p)

      # Predictions
     u_pred = self.R_data(self.x, self.t)
     r_data = u_pred - self.u
     #print(r_data)
     r_physic = self.R_physic(self.x, self.t)

     u_bc0d=tf.convert_to_tensor(0, dtype=tf.float32)
     u_bc1d= tf.convert_to_tensor(0, dtype=tf.float32)
     u_icd=tf.convert_to_tensor(np.sin(10*np.pi* self.x_ic), dtype=tf.float32)
     u_ic_td=tf.convert_to_tensor(0, dtype=tf.float32)
     u_bc0p=self.R_bc(self.x_bc0, self.t_bc0)
     u_bc1p=self.R_bc(self.x_bc1, self.t_bc1)
     u_icp=self.R_ic(self.x_ic, self.t_ic)

     with tf.GradientTape(persistent=True) as tape:
          tape.watch(self.t_ic)
          tape.watch(self.x_ic)


          u_icp=self.R_ic(self.x_ic, self.t_ic)

     u_ic_tp=tape.gradient(u_icp,self.t_ic)
     #print(u_ic_tp.shape)

     del tape

     r_bc0= u_bc0p- u_bc0d
     r_bc1=u_bc1p-u_bc1d
     r_ic=u_icp-u_icd
     #r_ic_t=u_ic_tp-u_ic_td


    # Combine residuals
     r = tf.concat([r_data, r_physic,r_bc0,r_bc1,r_ic], axis=0)
     #print(r.shape)
     loss = tf.reduce_sum(tf.square(r)).numpy()
     params = p[-1:]
     error1 = np.abs(params - 0.05)/0.05*100
     #error2 = np.abs(p2- 0.05)/0.05 * 100
     self.err_l1.append(error1)
     #self.err_l2.append(error2)

     self.loss_hist.append(loss)
     self.param_hist1.append(p[-1:])
     self.iteration_hist.append(self.iteration)

     if print_loss and self.iteration % 10 == 0:
        print(f"Iteration: {self.iteration}, Loss: {loss:.6f}, Parameters: {params}")

     self.iteration += 1
     return tf.reshape(r, [-1])
    def J(self, p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases, self.lambda_3])
            r = self.R(p, print_loss=False)
        jac_list = tape.jacobian(r, [*self.weights, *self.biases, self.lambda_3])


        del tape
        jac=np.hstack([tf.reshape(j, [r.shape[0], -1]).numpy() for j in jac_list])
       # print(jac)
        #print("_____________________________________________________")

        #jac=jac+l
        #print(jac)
        return jac

    def Rs(self, p):
        re = self.R(p, print_loss=True)
        return re.numpy()

    def train_trf(self):
        p = self.get_weights()
        print(p.shape)
        result = least_squares(
               self.Rs,
               p,
               jac=self.J,

              method='trf',
              #xtol=1e-6,      # Tolerance for change in x
        #ftol=1e-6,      # Tolerance for change in f
        #gtol=1e-6,
               #loss='cauchy',
      #f_scale=0.1,
       )
        self.set_weights(result.x)
        np.savez('heat.npz',
                 p1=self.param_hist1,
                 err1=self.err_l1,
                 loss_hist=self.loss_hist,
                 iteration_hist=self.iteration_hist)
        print("Optimization complete. Results saved to result.npz.")
        print("Optimized params:", result.x[-1:])

    def predict(self, X_star):
        X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
        x_star = X_star_tf[:, 0:1]
        t_star = X_star_tf[:, 1:2]
        u_star = self.R_data(x_star, t_star)
        return u_star.numpy()


def plot_results(x, t, Exact, u_pred):
    # Calculate the error
    error = np.abs(Exact - u_pred)
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot true data
    X, T = np.meshgrid(x, t)
    cp0 = axs[0].contourf(T, X, Exact, 20, cmap="rainbow")
    fig.colorbar(cp0, ax=axs[0])
    axs[0].set_title('True Solution')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')

    # Plot predicted data
    cp1 = axs[1].contourf(T, X, u_pred, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=axs[1])
    axs[1].set_title('Estimated Solution')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('x')

    # Plot error
    cp2 = axs[2].contourf(T, X, error, 20, cmap="rainbow")
    fig.colorbar(cp2, ax=axs[2])
    axs[2].set_title('|True - Estimated|')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('x')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nu = 0.01 / np.pi
    N_u = 500
    layers = [2,20,20, 1]
    x = np.linspace(0, 1, 200, dtype=np.float32)
    t = np.linspace(0, 1, 100, dtype=np.float32)
    
    X, T = np.meshgrid(x, t)
    beta = 1.0 / 20  # True value of beta from the article
   # Exact=np.sin(np.pi*X)*np.cos(a*np.pi*T)+0.5*np.sin(4*np.pi*X)*np.cos(4*a*np.pi*T)

    #Exact = np.sin(np.pi * X) * np.cos(np.pi**2 * T)
    Exact = np.exp(-(10 * np.pi * beta)**2 * T) * np.sin(10 * np.pi * X)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]



    #t=0
    X_ic=np.hstack((X[0,:].flatten()[:, None], T[0,:].flatten()[:, None]))
    #print(X_ic.shape[0])
    n_ic=200

    idx_ic = np.random.choice(X_ic.shape[0], n_ic, replace=False)
    X_train_ic = X_ic[idx_ic, :]
    #u_train = u_star[idx, :]

    n_bc=100
    #x=0
    X_bc0=np.hstack((X[:,0].flatten()[:, None], T[:,0].flatten()[:, None]))
    #print(X_bc0.shape)
    idx_bc0 = np.random.choice(X_bc0.shape[0], n_bc, replace=False)
    X_train_bc0 = X_bc0[idx_bc0, :]

    n_bc
    #x=1
    X_bc1=np.hstack((X[:,-1].flatten()[:, None], T[:,-1].flatten()[:, None]))

    idx_bc1 = np.random.choice(X_bc1.shape[0], n_bc, replace=False)
    X_train_bc1 = X_bc1[idx_bc1, :]

    model = RJ_PINNs(X_u_train, u_train, X_train_bc0,X_train_bc1,X_train_ic,layers,init_method='xavier',lambda_init_method='constant')
    model.train_trf()
    #p = model.get_weights()
    #model.J(p)

    u_pred= model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    u_pred = u_pred.reshape(Exact.shape)

    print('Error u: %e' % (error_u))

    # Plot results
    plot_results(x, t, Exact, u_pred)
