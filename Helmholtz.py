

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
    def __init__(self, X, u,X_0,X_1,Y_0, Y_1,layers,init_method='xavier',  lambda_init_method='constant'):

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.u = u

        # Define placeholders for inputs (compatible with TensorFlow v2)
        self.x = tf.convert_to_tensor(self.x, dtype=tf.float32)
        self.y = tf.convert_to_tensor(self.y, dtype=tf.float32)
        self.u = tf.convert_to_tensor(self.u, dtype=tf.float32)
        
        
        # Define placeholders for inputs (compatible with TensorFlow v2)
        self.xx_0 = tf.convert_to_tensor(X_0[:, 0:1], dtype=tf.float32)
        self.yx_0 = tf.convert_to_tensor(X_0[:, 1:2], dtype=tf.float32)




        self.xx_1 = tf.convert_to_tensor(X_1[:, 0:1], dtype=tf.float32)
        self.yx_1 = tf.convert_to_tensor(X_1[:, 1:2], dtype=tf.float32)





        self.xy_0 = tf.convert_to_tensor(Y_0[:, 0:1], dtype=tf.float32)
        self.yy_0 = tf.convert_to_tensor(Y_0[:, 1:2], dtype=tf.float32)

        self.xy_1 = tf.convert_to_tensor(Y_1[:, 0:1], dtype=tf.float32)
        self.yy_1 = tf.convert_to_tensor(Y_1[:, 1:2], dtype=tf.float32)


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

    def R_data(self, x, y):
        X = tf.concat([x, y], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u
    
    def R_bc(self, x, y):
        X = tf.concat([x, y], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u
    
    def R_ic(self, x, y):
        X = tf.concat([x, y], axis=1)
        u = self.neural_net(X, self.weights, self.biases)
        return u

    def R_physic(self, x, y):
        lambda_3 = self.lambda_3
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u = self.R_data(x, y)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
        u_xx = tape.gradient(u_x, x)
            #u_xxx = tape.gradient(u_xx, x)
        u_yy = tape.gradient(u_y, y)

        #u_xxxx = tape.gradient(u_xxx, x)
        del tape #deleted the tape after it is used

        
        f =- u_xx - u_yy-lambda_2**2*u-lambda_2**2*tf.sin(lambda_2*x)*tf.sin(lambda_2*y)
        return f


    def get_weights(self):
        variables = [*self.weights, *self.biases,  self.lambda_2]
        return tf.concat([tf.reshape(var, [-1]) for var in variables], axis=0)

    def set_weights(self, flat_weights):
        idx = 0
        for var in [*self.weights, *self.biases, self.lambda_2]:
            shape = tf.shape(var)
            size = tf.reduce_prod(shape)
            new_values = tf.reshape(flat_weights[idx:idx + size], shape)
            new_values = tf.cast(new_values, dtype=tf.float32)
            var.assign(new_values)
            idx += size

    def R(self, p, print_loss=True):
     self.set_weights(p)

      # Predictions
     u_pred = self.R_data(self.x, self.y)
     r_data = u_pred - self.u
     #print(r_data)
     r_physic = self.R_physic(self.x, self.y)

     ux0d=tf.convert_to_tensor(0, dtype=tf.float32)
     ux1d= tf.convert_to_tensor(0, dtype=tf.float32)
     uy0d=tf.convert_to_tensor(0, dtype=tf.float32)

     uy1d=tf.convert_to_tensor(0, dtype=tf.float32)
     ux0p=self.R_bc(self.xx_0, self.yx_0)
     ux1p=self.R_bc(self.xx_1, self.yx_1)
     #u_bc1p=self.net_u(self.x_bc1, self.t_bc1)

     uy0p=self.R_bc(self.xy_0, self.yy_0)
     uy1p=self.R_bc(self.xy_1, self.yy_1)





     rx0= ux0p- ux0d
     rx1=ux1p-ux1d
     ry0=uy0p-uy0d
     ry1= uy1p- uy1d
     #r_bc0_xx=u_bc0_xxp-u_bc0_xxd
     #r_bc1_xx=u_bc1_xxp-u_bc1_xxd

    # Combine residuals
     r = tf.concat([r_data, r_physic], axis=0)
     #print(r.shape)
     loss = tf.reduce_sum(tf.square(r)).numpy()
     params = p[-1:]
     #error1 = np.abs(params - 1)/100
     #error2 = np.abs(p2- 0.05)/0.05 * 100
     #self.err_l1.append(error1)
     #self.err_l2.append(error2)

     #self.loss_hist.append(loss)
     #self.param_hist1.append(p[-1:])
     #self.iteration_hist.append(self.iteration)

     if print_loss and self.iteration % 10 == 0:
        print(f"Iteration: {self.iteration}, Loss: {loss:.6f}, Parameters: {params}")

        #print(f"error1:{error1}")
     self.iteration += 1
     return tf.reshape(r, [-1])
    def J(self, p):
        self.set_weights(p)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([*self.weights, *self.biases,  self.lambda_2])
            r = self.R(p, print_loss=False)
        jac_list = tape.jacobian(r, [*self.weights, *self.biases, self.lambda_2])


        del tape
        jac=np.hstack([tf.reshape(j, [r.shape[0], -1]).numpy() for j in jac_list])
       # print(jac)
        #print("_____________________________________________________")

        l=np.array([1e-14])
        #jac=jac+l
        #print(jac)
        return jac

    def Rs(self, p):
        re = self.R(p, print_loss=True)
        return re.numpy()

    def train_trf(self):
      p = self.get_weights()
      print(p.shape)

      # Calculate number of parameters
      total_params = len(p)
      num_lambda_params = 3  # lambda_1, lambda_2, lambda_3

    # Set bounds:
    # - No bounds for neural network weights/biases (-inf to inf)
    # - Custom bounds for lambda parameters
      lower_bounds = [-np.inf] * (total_params - num_lambda_params) + [0.1, 0.1, 0.1]  # Example: lambda >= 0.1
      upper_bounds = [np.inf] * (total_params - num_lambda_params) + [10.0, 10.0, 10.0]  # Example: lambda <= 10.0

      bounds = (lower_bounds, upper_bounds)

      result = least_squares(
        self.Rs,
        p,
        jac=self.J,
        #bounds=bounds,
        method='trf',  # Trust Region Reflective algorithm (supports bounds)
       # max_nfev=500,  # Maximum number of function evaluations
       # verbose=2      # Show optimization progress
    )

      self.set_weights(result.x)
      print("Optimized lambda parameters:", result.x[-1:])

    def predict(self, X_star):
        X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)
        x_star = X_star_tf[:, 0:1]
        t_star = X_star_tf[:, 1:2]
        u_star = self.R_data(x_star, t_star)
        return u_star.numpy()


# Remplacez la fonction plot_results par ceci :
def plot_specific_slices(x, y, Exact, u_pred):
    plt.figure(figsize=(12, 5))

    # Plot for Y=1
    plt.subplot(1, 2, 1)
    plt.plot(x, Exact[-1, :], 'b-', linewidth=4, label='Exact')
    plt.plot(x, u_pred[-1, :], 'r--', linewidth=4, label='Estimated')
    plt.xlabel('x')
    plt.ylabel('u(x,y=1)')
    plt.title('Solution at y=1')
    plt.legend()

    # Plot for X=1
    plt.subplot(1, 2, 2)
    plt.plot(y, Exact[:, -1], 'b-', linewidth=4, label='Exact')
    plt.plot(y, u_pred[:, -1], 'r--', linewidth=4, label='Estimated')
    plt.xlabel('y')
    plt.ylabel('u(x=1,y)')
    plt.title('Solution at x=1')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Et dans la partie main, remplacez l'appel à plot_results par :


if __name__ == "__main__":
    N_u = 1000
    n=2
    k=np.sqrt(n*np.pi**2+1)#true value
    #k=2*np.pi*n
    layers = [2, 20,20,20,1]

    x = np.linspace(0, 1, 200, dtype=np.float32)
    y = np.linspace(0, 1, 100, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    Exact = np.sin(k*X)*np.sin(k*Y)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    
    id = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[id, :]
    u_train = u_star[id, :]



    #t=0
    Y_0=np.hstack((X[0,:].flatten()[:, None], Y[0,:].flatten()[:, None]))
    #print(X_ic.shape[0])
    n_ic=100

    idy0 = np.random.choice(Y_0.shape[0], n_ic, replace=False)
    Y_train_0 = Y_0[idy0, :]
    #u_train = u_star[idx, :]


    Y_1=np.hstack((X[-1,:].flatten()[:, None], Y[-1,:].flatten()[:, None]))
    #print(X_ic.shape[0])
    #n_ic=100

    idy1 = np.random.choice(Y_1.shape[0], n_ic, replace=False)
    Y_train_1 = Y_1[idy1, :]
    #u_train = u_star[idx, :]

    n_bc=100
    #x=0
    X_0=np.hstack((X[:,0].flatten()[:, None], Y[:,0].flatten()[:, None]))
    #print(X_bc0.shape)
    idx0 = np.random.choice(X_0.shape[0], n_bc, replace=False)
    X_train_0 = X_0[idx0, :]

    n_bc
    #x=1
    X_1=np.hstack((X[:,-1].flatten()[:, None], Y[:,-1].flatten()[:, None]))

    idx1 = np.random.choice(X_1.shape[0], n_bc, replace=False)
    X_train_1 = X_1[idx1, :]


    model = RJ_PINNs(X_u_train, u_train, X_train_0,X_train_1,Y_train_0,Y_train_1,layers,init_method='xavier',lambda_init_method='constant')
    model.train_trf()
    #p = model.get_weights()
    #model.J(p)

    u_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    u_pred = u_pred.reshape(Exact.shape)

    print('Error u: %e' % (error_u))

    # Plot results
    # As 't' was not defined, assuming it should be 'y' for this 2D problem
    #plot_results(x, y, Exact, u_pred)

    print('Error u: %e' % (error_u))

    # Plot results
    plot_specific_slices(x, y, Exact, u_pred)
