
import numpy as np
from sympy import symbols, cos, sin, simplify
from scipy.optimize import minimize_scalar

x, y = symbols('x y')

det_H = simplify((cos(x) * cos(y/10)) * (cos(x) * cos(y/10) / 100) - (-1/10 * sin(x) * sin(y/10))**2)

det_H_simplified = simplify(det_H)

print(det_H_simplified)



def f(x):
    return -np.cos(x[0]) * np.cos(x[1] / 10)


def grad_f(x):
    return np.array([
        np.sin(x[0]) * np.cos(x[1] / 10),
        1/10 * np.cos(x[0]) * np.sin(x[1] / 10)
    ])


def steepest_descent_no_line_search(x0, max_iter=100, tol=1e-6, step_size=0.01):
    x = x0
    grad0_norm = np.linalg.norm(grad_f(x))
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - step_size * grad
        grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
        print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}")
        if np.linalg.norm(grad) < tol:
            print("Convergence achieved.")
            break
    return x


def exact_line_search(f, x, p):
    func = lambda alpha: f(x + alpha * p)
    result = minimize_scalar(func)
    return result.x


def steepest_descent_exact_line_search(x0, max_iter=100, tol=1e-6):
    x = x0
    grad0_norm = np.linalg.norm(grad_f(x))
    for i in range(max_iter):
        grad = grad_f(x)
        alpha = exact_line_search(f, x, -grad)
        x = x - alpha * grad
        grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
        print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}")
        if np.linalg.norm(grad) < tol:
            print("Convergence achieved.")
            break
    return x


def armijo_line_search(f, grad_f, x, p, alpha=1.0, beta=0.5, sigma=0.1):
    while f(x + alpha * p) > f(x) + sigma * alpha * np.dot(grad_f(x), p):
        alpha *= beta
    return alpha


def steepest_descent_armijo(x0, max_iter=100, tol=1e-6):
    x = x0
    grad0_norm = np.linalg.norm(grad_f(x))
    for i in range(max_iter):
        grad = grad_f(x)
        p = -grad
        alpha = armijo_line_search(f, grad_f, x, p)
        x = x + alpha * p
        grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
        print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}")
        if np.linalg.norm(grad) < tol:
            print("Convergence achieved.")
            break
    return x



initial_conditions = [
    np.array([-np.pi/4, np.pi/4]),
    np.array([np.pi/4, -np.pi/4]),
    np.array([0, 0])
]

results_no_line_search = [steepest_descent_no_line_search(x0, max_iter=25, tol=1e-3, step_size=0.01) for x0 in initial_conditions]
results_exact_line_search = [steepest_descent_exact_line_search(x0, max_iter=25, tol=1e-3) for x0 in initial_conditions]
results_armijo = [steepest_descent_armijo(x0, max_iter=25, tol=1e-3) for x0 in initial_conditions]

print("\n")
print("Steepest Descent without Line Search:")
print(results_no_line_search)

print("\n")
print("\nSteepest Descent with Exact Line Search:")
print(results_exact_line_search) 

print("\n")
print("\nSteepest Descent with Armijo Line Search:")
print(results_armijo)




def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
                      200 * (x[1] - x[0]**2)])


def steepest_descent(f, grad_f, x0, line_search=None, max_iter=100, tol=1e-6):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        if line_search == 'exact':
            alpha = exact_line_search(f, x, -grad)
        elif line_search == 'armijo':
            alpha = armijo_line_search(f, grad_f, x, -grad)
        else:  
            alpha = 0.001
        x = x - alpha * grad
        if np.linalg.norm(grad) < tol:
            print(f"Convergence after {i+1} iterations.")
            break
        print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}")
    return x

x0 = np.array([-1, 1])

print("Without line search:")
steepest_descent(rosenbrock, grad_rosenbrock, x0, max_iter=50)

print("\nWith Armijo line search:")
steepest_descent(rosenbrock, grad_rosenbrock, x0, line_search='armijo', max_iter=50)