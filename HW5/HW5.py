# import numpy as np

# def f(x):
#     return -np.cos(x[0]) * np.cos(x[1] / 10)

# def grad_f(x):
#     return np.array([
#         np.sin(x[0]) * np.cos(x[1] / 10),
#         1/10 * np.cos(x[0]) * np.sin(x[1] / 10)
#     ])

# def armijo_line_search(f, grad_f, x, p, alpha=1.0, beta=0.5, sigma=0.1):
#     while f(x + alpha * p) > f(x) + sigma * alpha * np.dot(grad_f(x), p):
#         alpha *= beta
#     return alpha

# def steepest_descent_armijo(x0, max_iter=100, tol=1e-6):
#     x = x0
#     grad0_norm = np.linalg.norm(grad_f(x0))
#     for i in range(max_iter):
#         grad = grad_f(x)
#         p = -grad
#         alpha = armijo_line_search(f, grad_f, x, p)
#         x = x + alpha * p
#         grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
#         print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}, Step Size = {alpha}")
#         if np.linalg.norm(grad) < tol:
#             print("Convergence achieved.")
#             break


# def hessian_f(x):
#     return np.array([
#         [-np.cos(x[0]) * np.cos(x[1] / 10), 0],
#         [0, -1/100 * np.cos(x[0]) * np.cos(x[1] / 10)]
#     ])

# def newtons_method_armijo(x0, max_iter=100, tol=1e-6):
#     x = x0
#     grad0_norm = np.linalg.norm(grad_f(x0))
#     for i in range(max_iter):
#         grad = grad_f(x)
#         H = hessian_f(x)
#         p = -np.linalg.solve(H, grad)
#         alpha = armijo_line_search(f, grad_f, x, p)
#         x = x + alpha * p
#         grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
#         print(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}, Step Size = {alpha}")
#         if np.linalg.norm(grad) < tol:
#             print("Convergence achieved.")
#             break


# It seems the required libraries and functions were not defined within the code execution environment.
# Let's define the necessary imports, functions, and then run the algorithms.


import numpy as np

def f(x):
    return -np.cos(x[0]) * np.cos(x[1] / 10)

def grad_f(x):
    return np.array([
        np.sin(x[0]) * np.cos(x[1] / 10),
        1/10 * np.cos(x[0]) * np.sin(x[1] / 10)
    ])

def hessian_f(x):
    return np.array([
        [-np.cos(x[0]) * np.cos(x[1] / 10), 0],
        [0, -1/100 * np.cos(x[0]) * np.cos(x[1] / 10)]
    ])

def armijo_line_search(f, grad_f, x, p, alpha=1.0, beta=0.5, sigma=0.1):
    while f(x + alpha * p) > f(x) + sigma * alpha * np.dot(grad_f(x), p):
        alpha *= beta
    return alpha

def steepest_descent_armijo(x0, max_iter=100, tol=1e-6):
    x = x0
    grad0_norm = np.linalg.norm(grad_f(x0))
    print_output = []
    for i in range(max_iter):
        grad = grad_f(x)
        p = -grad
        alpha = armijo_line_search(f, grad_f, x, p)
        x = x + alpha * p
        grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
        print_output.append(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}, Step Size = {alpha}")
        if np.linalg.norm(grad) < tol:
            print_output.append("Convergence achieved.")
            break
    return print_output

def newtons_method_armijo(x0, max_iter=100, tol=1e-6):
    x = x0
    grad0_norm = np.linalg.norm(grad_f(x0))
    print_output = []
    for i in range(max_iter):
        grad = grad_f(x)
        H = hessian_f(x)
        p = -np.linalg.solve(H, grad)
        alpha = armijo_line_search(f, grad_f, x, p)
        x = x + alpha * p
        grad_norm_reduction = np.linalg.norm(grad) / grad0_norm
        print_output.append(f"Iter {i+1}: x = {x}, f(x) = {f(x)}, Grad Norm Reduction = {grad_norm_reduction}, Step Size = {alpha}")
        if np.linalg.norm(grad) < tol:
            print_output.append("Convergence achieved.")
            break
    return print_output

x0 = np.array([0.5, -0.5])

# sd_results = steepest_descent_armijo(x0, max_iter=50, tol=1e-3)

# print("Steepest Descent with Armijo Line Search Results:")
# for result in sd_results:
#     print(result)


newton_results = newtons_method_armijo(x0, max_iter=50, tol=1e-6)
print("Newton's Method with Armijo Line Search Results:")
for result in newton_results:
    print(result)