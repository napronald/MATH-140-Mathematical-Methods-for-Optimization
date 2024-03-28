import numpy as np
from scipy.optimize import line_search
from scipy.linalg import cho_solve, cho_factor
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def steepest_descent_method(x0, tol=1e-6, max_iter=100):
    xk = x0

    for i in range(max_iter):
        grad = grad_f(xk)
        pk = -grad 

        ls_res = line_search(f, grad_f, xk, pk)
        alpha_k = ls_res[0]

        xk1 = xk + alpha_k * pk

        if np.linalg.norm(grad) < tol:
            print(f'Convergence achieved after {i} iterations.')
            break

        print(f"Iteration: {i}, xk: [{xk1[0]:.5f}, {xk1[1]:.5f}], f(xk): {f(xk1):.5f}, Gradient Norm: {np.linalg.norm(grad):.5f}")

        xk = xk1

    return xk

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




def f(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_f(x):
    dfdx1 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dfdx2 = 200*(x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])

def hess_f(x):
    d2fdx12 = 2 - 400*x[1] + 1200*x[0]**2
    d2fdx22 = 200
    d2fdxdy = -400*x[0]
    return np.array([[d2fdx12, d2fdxdy], [d2fdxdy, d2fdx22]])


def solve_trust_region_subproblem(g, B, Delta, trsubtol=1e-6):
    try:
        p = np.linalg.solve(B, -g)
        if norm(p) <= Delta:
            return p
    except np.linalg.LinAlgError:
        pass

    lambda_ = 0.001
    while True:
        try:
            B_lambda = B + lambda_ * np.eye(len(B))
            factor = cho_factor(B_lambda)
            p = cho_solve(factor, -g)
            u = cho_solve(factor, p)

            if norm(p) <= Delta + trsubtol:
                break

            lambda_ += (norm(p) / norm(u))**2 * ((norm(p) - Delta) / Delta)
        except np.linalg.LinAlgError:
            lambda_ *= 10 
    
    return p


def trust_region_method(x0, k_max=100, gtol=1e-9, mu=0.25, mu_e=0.75):
    xk = x0
    Delta_k = 1.0
    k = 0
    x_true = np.array([1, 1])
    iter_points = [xk]
    deltas = [Delta_k]
    prev_error = None

    while k < k_max:
        gk = grad_f(xk)
        Bk = hess_f(xk)
        fk = f(xk)
        error = norm(xk - x_true)
        error_reduction = "-" if prev_error is None else f"{error / prev_error:.5f}"
        prev_error = error

        if norm(gk) < gtol:
            break

        pk = solve_trust_region_subproblem(gk, Bk, Delta_k, 1e-6)
        fkp = f(xk + pk)
        mk0 = fk
        mkp = fk + np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk))
        rho_k = (fk - fkp) / (mk0 - mkp)

        print(f"Iter: {k}, xk: [{xk[0]:.5f}, {xk[1]:.5f}], pk: [{pk[0]:.5f}, {pk[1]:.5f}], Delta_k: {Delta_k:.5f}, error: {error:.5f}, error reduction: {error_reduction}")

        if rho_k > mu:
            xk = xk + pk
            iter_points.append(xk)
            deltas.append(Delta_k)
            if rho_k >= mu_e:
                Delta_k = max(Delta_k, 2*norm(pk))
        else:
            Delta_k = Delta_k / 2

        k += 1

    return xk, k, iter_points, deltas

# lower_bounds = np.array([-np.pi/2, -10*np.pi/2])
# upper_bounds = np.array([np.pi/2, 10*np.pi/2])

# np.random.seed(1)
# x0 = np.random.uniform(low=lower_bounds, high=upper_bounds)

x0 = np.array([-1, 1])
print(f"Initial Guess: [{x0[0]:.5f}, {x0[1]:.5f}]")

print(f"Trust Region Method")
x_min, iter, iter_points, deltas = trust_region_method(x0)
print(f"Minimizer: [{x_min[0]:.5f}, {x_min[1]:.5f}] after {iter} iterations.")


print(f"Steepest Descent Method")
x_min_steep = steepest_descent_method(x0)
print(f"Minimizer: [{x_min_steep[0]:.5f}, {x_min_steep[1]:.5f}]")

print(f"Newton's Method")
x_min = newton_method_line_search(x0)
print(f"Minimizer: [{x_min[0]:.5f}, {x_min[1]:.5f}]")

print(f"BFGS Method")
x_min = bfgs_quasi_newton(x0)
print(f"Minimizer: [{x_min[0]:.5f}, {x_min[1]:.5f}]")



x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(np.array([X, Y]))
x_true = np.array([1, 1])


x = np.linspace(-2, 2, 800)
y = np.linspace(-1, 3, 800)
X, Y = np.meshgrid(x, y)
Z = f(np.array([X, Y]))
x_true = np.array([1, 1])

fig, ax = plt.subplots(figsize=(12, 8))
CS = ax.contour(X, Y, Z, levels=np.logspace(-2, 5, 100), cmap='viridis')
ax.clabel(CS, inline=1, fontsize=8)
ax.plot(*x_true, 'r*', markersize=15, label='True Minimum')
ax.plot([p[0] for p in iter_points], [p[1] for p in iter_points], 'ro-', label='Iterations')

for point, delta in zip(iter_points, deltas):
    circle = Circle(point, delta, color='blue', fill=False, linestyle='-', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)

ax.text(x0[0], x0[1], 'Start', fontsize=20, ha='right')
ax.text(x_min[0], x_min[1], 'End', fontsize=20, ha='right')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Objective Function Contour and Trust Region Iterations')
ax.legend()
plt.show()