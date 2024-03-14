import numpy as np
from scipy.optimize import line_search

def f(x):
    return -np.cos(x[0]) * np.cos(x[1] / 10)

def grad_f(x):
    dfdx = np.sin(x[0]) * np.cos(x[1] / 10)
    dfdy = np.cos(x[0]) * np.sin(x[1] / 10) / 10
    return np.array([dfdx, dfdy])

def hess_f(x):
    d2fdx2 = np.cos(x[0]) * np.cos(x[1] / 10)
    d2fdxdy = -np.sin(x[0]) * np.sin(x[1] / 10) / 10
    d2fdydx = -np.sin(x[0]) * np.sin(x[1] / 10) / 10
    d2fdy2 = np.cos(x[0]) * np.cos(x[1] / 10) / 100
    return np.array([[d2fdx2, d2fdxdy], [d2fdydx, d2fdy2]])

def bfgs_quasi_newton(x0, tol=1e-6, max_iter=100):
    xk = x0
    Hk = np.eye(len(x0))  

    for i in range(max_iter):
        grad = grad_f(xk)
        pk = -np.dot(Hk, grad)  

        ls_res = line_search(f, grad_f, xk, pk)
        alpha_k = ls_res[0]

        xk1 = xk + alpha_k * pk

        sk = xk1 - xk
        yk = grad_f(xk1) - grad

        if np.linalg.norm(grad) < tol:
            print(f'Convergence achieved after {i} iterations.')
            break

        rho_k = 1.0 / (np.dot(yk, sk))
        I = np.eye(len(x0))
        Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        print(f"Iteration: {i}, xk: [{xk1[0]:.5f}, {xk1[1]:.5f}], f(xk): {f(xk1):.5f}, Gradient Norm Reduction: {np.linalg.norm(grad_f(xk1))/np.linalg.norm(grad_f(x0)):.5f}, Alpha: {alpha_k:.5f}")

        xk = xk1

    return xk

def newton_method_line_search(x0, tol=1e-6, max_iter=100):
    xk = x0

    for i in range(max_iter):
        grad = grad_f(xk)
        Hk = hess_f(xk)
        pk = -np.linalg.solve(Hk, grad) 

        ls_res = line_search(f, grad_f, xk, pk)
        alpha_k = ls_res[0]

        xk1 = xk + alpha_k * pk

        if np.linalg.norm(grad) < tol:
            print(f'Convergence achieved after {i} iterations.')
            break

        print(f"Iteration: {i}, xk: [{xk1[0]:.5f}, {xk1[1]:.5f}], f(xk): {f(xk1):.5f}, Gradient Norm Reduction: {np.linalg.norm(grad_f(xk1))/np.linalg.norm(grad_f(x0)):.5f}, Alpha: {alpha_k:.5f}")

        xk = xk1

    return xk

lower_bounds = np.array([-np.pi/2, -10*np.pi/2])
upper_bounds = np.array([np.pi/2, 10*np.pi/2])

np.random.seed(3) # Used seed 0, 1, 3
x0 = np.random.uniform(low=lower_bounds, high=upper_bounds)
print(f"Initial Guess: [{x0[0]:.5f}, {x0[1]:.5f}]")

print(f"Newton's Method")
x_min = newton_method_line_search(x0)
print(f"Minimizer: [{x_min[0]:.5f}, {x_min[1]:.5f}]")

print(f"BFGS Method")
x_min = bfgs_quasi_newton(x0)
print(f"Minimizer: [{x_min[0]:.5f}, {x_min[1]:.5f}]")
