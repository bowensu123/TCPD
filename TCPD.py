import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly.decomposition import parafac
from tensorly import tensor
from scipy import io
import time 
import os
def Thres(X, zeta):
    return X * (np.abs(X) > zeta)


def TCPD(D, r, para={}):
    d = D.shape
    n = D.ndim
    I = [None] * n
    max_iter = para.get('max_iter', 200)
    epsilon = para.get('epsilon', 1e-6)
    zeta = para.get('zeta', 1.1 * np.max(D))
    gamma = para.get('gamma', 0.7)
    CI = para.get('CI', 2)
    
    D_sub_mat = [None] * n
    S_sub_mat = [None] * n
    L_sub_mat = [None] * n
    X_sub_mat = [None] * n
    err = -1 * np.ones(max_iter)
    timer = -1 * np.ones(max_iter)
    
    len_dim = np.zeros(n, dtype=int)  # Length for each mode in L_core

    # Default/Inputted parameters
    for i in range(n):
        len_dim[i] = min(int(np.ceil(CI * r[i] * np.log(d[i]))), d[i])

    # Initialize L_sub_mat and L_core
    for i in range(n):
        shapeL = list(len_dim)
        shapeL[i] = d[i]
        L_sub_mat[i] = tl.tensor(np.zeros(shapeL))  
        L_core = tl.tensor(np.zeros(len_dim))        
        I[i] = np.random.permutation(d[i])[:len_dim[i]]  
    norm_of_D = 0
    D_core = D[np.ix_(*I)]  
    norm_of_D += np.linalg.norm(D_core) ** 2  

    for i in range(n):
        J = I[:] 
        J[i] = np.arange(d[i])  
        D_sub_mat[i] = D[np.ix_(*J)]  
        norm_of_D += np.linalg.norm(D_sub_mat[i]) ** 2  
    for it in range(max_iter):
        start_time = time.time()

        # Compute chidori and core for L^(k+1)
        if it > 0:
            for i in range(n):
                L_sub_mat[i] = L_core.copy()
                for j in range(n):
                    if j == i:
                        L_sub_mat[i] = mode_dot(L_sub_mat[i], X_sub_mat[j], j)
                    else:
                        L_sub_mat[i] = mode_dot(L_sub_mat[i], X_sub_mat[j][I[j], :], j)
            for i in range(n):
                L_core = mode_dot(L_core, X_sub_mat[i][I[i], :], i)
        error = 0
        for i in range(n):
            S_sub_mat[i] = Thres(D_sub_mat[i] - L_sub_mat[i], zeta)
            error += np.linalg.norm(D_sub_mat[i] - L_sub_mat[i] - S_sub_mat[i]) ** 2

        S_core = Thres(D_core - L_core, zeta)
        error += np.linalg.norm(D_core - S_core - L_core) ** 2
        L_core = D_core - S_core
        for i in range(n):
            C = tl.unfold(D_sub_mat[i] - S_sub_mat[i], i)
            Q, R = np.linalg.qr(C[I[i], :].T)  
            R_trunc = R[:r, :]  
            Q_trunc = Q[:, :r]  
            R_inv = np.linalg.pinv(R_trunc)  

            X_sub_mat[i] = C @ Q_trunc @ R_inv.T

        zeta *= gamma
        timer[it] = time.time() - start_time
        err[it] = np.sqrt(error / norm_of_D)

        if err[it] <= epsilon:
            if 'info' in para:
                print(f"Total {it + 1} iterations, final error: {err[it]:e}, total time: {np.sum(timer[timer > 0]):f}")
            return L_core, X_sub_mat, timer, err

    print("Process completed at max_iter")
    return L_core, X_sub_mat, timer, err
