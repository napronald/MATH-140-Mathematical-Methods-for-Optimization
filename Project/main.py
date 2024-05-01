import time
import scipy
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--func', type=str, default='quad', choices=['quad', 'rosen'], help='Specify function')
parser.add_argument('--n_dim', type=int, default=2, help='Dimension size')
parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--reg', type=int, default=1e9, help='Regularization factor')
args = parser.parse_args()

device = torch.device("cpu") 
torch.manual_seed(args.seed)

x0 = torch.rand(args.n_dim, device=device, dtype=torch.float64)
print(f"Initial Guess: {x0}\n")


if args.func == "quad":
    A = torch.rand(args.n_dim, args.n_dim, device=device, dtype=torch.float64)
    A = torch.mm(A, A.t()) + torch.eye(args.n_dim, device=device, dtype=torch.float64) * args.reg
    b = torch.rand(args.n_dim, device=device, dtype=torch.float64)
    c = torch.rand(1, device=device, dtype=torch.float64)
    def f(x):
        return 0.5 * torch.dot(x, torch.mv(A, x)) - torch.dot(b, x) + c

    def grad_f(x):
        return torch.mv(A, x) - b

    def hess_f(x):
        return A

    x_true = torch.linalg.solve(A, b)
    condition_number = torch.linalg.cond(A) # Adjust args.reg to make more or less well-conditioned
    expected_relative_error = condition_number * 2.22e-16
    digits_of_accuracy = -torch.log10(expected_relative_error)

    print(f"Condition_number: {condition_number}")
    print(f"Expected relative error: {expected_relative_error}")
    print(f"Digits of accuracy: {digits_of_accuracy}\n")
    print("Analytical minimum x:", x_true)
    print("Gradient at analytical minimum:", grad_f(x_true))
    print("Function value at analytical minimum:", f(x_true))
    print()

elif args.func == "rosen":
    def f(x):
        return torch.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def grad_f(x):
        n = x.size(0)
        grad = torch.zeros_like(x)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        for i in range(1, n-1):
            grad[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        if n > 1:
            grad[-1] = 200 * (x[-1] - x[-2]**2)
        return grad

    def hess_f(x):
        n = x.size(0)
        hessian = torch.zeros((n, n))
        for i in range(n):
            if i < n - 1:
                hessian[i, i] = 1200 * x[i]**2 - 400 * x[i+1] + 2
                hessian[i, i+1] = -400 * x[i]
            if i > 0:
                hessian[i, i] += 200
                hessian[i, i-1] = -400 * x[i-1]
        return hessian

    x_true = torch.ones(args.n_dim, dtype=torch.float64, device=device)



def conjugate_gradient(A, b, x0, max_iter=50, tol=1e-6, Delta=None):
    x = x0.clone()
    r = b.double() - torch.mv(A.double(), x.double())
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = torch.mv(A.to(torch.float64), p)
        alpha = rs_old / torch.dot(p, Ap)
        x_new = x + alpha * p

        # For trust region 
        if Delta is not None:
            if torch.norm(x_new) >= Delta:
                alpha_max = -torch.dot(x.double(), p) + torch.sqrt((torch.dot(x.double(), p))**2 + torch.dot(p, p) * (Delta**2 - torch.dot(x.double(), x.double())))
                alpha_max /= torch.dot(p, p)
                x = x + alpha_max * p
                break

        x = x_new
        r -= alpha * Ap

        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, i


def line_search(f, grad_f, x, p, alpha=1.0, beta=0.5, sigma=0.1):
    while f(x + alpha * p) > f(x) + sigma * alpha * torch.dot(grad_f(x), p):
        alpha *= beta
        if alpha < 1e-10:
            break
    return alpha


def steepest_descent(x0, max_iter=50, tol=1e-6):
    xk = x0.clone()

    for i in range(max_iter):
        grad = grad_f(xk)
        pk = -grad

        alpha_k = line_search(f, grad_f, xk, pk)
        xk += alpha_k * pk

        if torch.norm(grad) < tol:
            break

    return xk, i + 1


def newton_method(x0, max_iter=50, tol=1e-6):
    xk = x0.clone()

    for i in range(max_iter):
        grad = grad_f(xk)
        Hk = hess_f(xk)
        pk = -torch.linalg.solve(Hk.cpu().to(torch.float64), grad.cpu().to(torch.float64))

        alpha_k = line_search(f, grad_f, xk, pk.to(device))
    
        xk += alpha_k * pk.to(device)

        if torch.norm(grad) < tol:
            break

    return xk, i + 1


def newton_method_cg(x0, max_iter=50, tol=1e-6):
    xk = x0.clone()

    for i in range(max_iter):
        grad = grad_f(xk)
        Hk = hess_f(xk)
        pk, i = conjugate_gradient(Hk.to(torch.float64), -grad.to(torch.float64), xk, max_iter, tol) 
        alpha_k = line_search(f, grad_f, xk, pk.to(device))
    
        xk += alpha_k * pk.to(device)

        if torch.norm(grad) < tol:
            break

    return xk, i + 1


def bfgs_quasi_newton(x0, max_iter=50, tol=1e-6):
    device = x0.device
    dtype = torch.float64  

    xk = x0.clone().detach().to(dtype=dtype)
    Hk = torch.eye(len(x0), device=device, dtype=dtype)

    for i in range(max_iter):
        grad = grad_f(xk).to(dtype=dtype) 
        current_norm = torch.norm(grad, p=2)

        if current_norm < tol:
            return xk, i + 1

        pk = -torch.mv(Hk, grad)
        alpha_k = line_search(f, grad_f, xk, pk)
        xk1 = xk + alpha_k * pk
        sk = xk1 - xk
        grad_new = grad_f(xk1).to(dtype=dtype)
        yk = grad_new - grad

        if not torch.all(torch.isfinite(yk)) or not torch.all(torch.isfinite(sk)):
            return xk, i + 1

        rho = 1.0 / torch.dot(yk, sk) if torch.dot(yk, sk) != 0 else 1e-6

        I = torch.eye(len(x0), device=device, dtype=dtype)
        Hk = (I - rho * torch.ger(sk, yk)) @ Hk @ (I - rho * torch.ger(yk, sk)) + rho * torch.ger(sk, sk)

        xk = xk1

    return xk, i + 1


def cg_subproblem(g, B, Delta, max_iter=50, tol=1e-6):
    x0 = torch.zeros(g.shape[0], device=device)
    p, i = conjugate_gradient(B, -g, x0, max_iter, tol, Delta) 
    return p


def direct_subproblem(g, B, Delta, trsubtol=1e-6):
    device = g.device
    n = len(B)
    I = torch.eye(n, dtype=torch.float64, device=device)

    try:
        L = torch.linalg.cholesky(B)
        p = torch.cholesky_solve(-g.unsqueeze(1), L).squeeze(1)
        if torch.norm(p) <= Delta:
            return p
    except RuntimeError:
        pass  

    lambda_ = torch.tensor(0.1, dtype=torch.float64, device=device)
    max_lambda = torch.tensor(1e10, dtype=torch.float64, device=device)
    min_lambda = torch.tensor(1e-4, dtype=torch.float64, device=device)

    B_lambda = B.clone()
    while True:
        B_lambda += lambda_ * I  
        try:
            L = torch.linalg.cholesky(B_lambda)
            p = torch.cholesky_solve(-g.unsqueeze(1), L).squeeze(1)
            p_norm = torch.norm(p)

            if p_norm <= Delta + trsubtol:
                break

            if p_norm > Delta:
                lambda_ *= 1.5
            else:
                lambda_ /= 2

            lambda_ = torch.min(torch.max(lambda_, min_lambda), max_lambda)

        except RuntimeError:
            lambda_ *= 10
            if lambda_ > max_lambda:
                return torch.zeros_like(g)  

    return p


def trust_region(x0, max_iter=50, gtol=1e-6, mu=0.25, mu_e=0.75, sub=None):
    device = x0.device
    xk = x0.clone().detach()
    Delta_k = 1.0
    k = 0

    while k < max_iter:
        gk = grad_f(xk).to(device)
        Bk = hess_f(xk).to(device)
        fk = f(xk).to(device)

        if torch.norm(gk) < gtol:
            break

        if sub == 'cg':
            pk = cg_subproblem(gk, Bk, Delta_k)
        elif sub == 'direct':
            pk = direct_subproblem(gk, Bk, Delta_k)

        fkp = f(xk + pk)
        mk0 = fk
        mkp = fk + torch.dot(gk, pk) + 0.5 * torch.dot(pk, torch.mv(Bk.to(torch.float64), pk.to(torch.float64)))
        denominator = (mk0 - mkp)
        rho_k = (fk - fkp) / denominator if abs(denominator) > 1e-6 else 0.0

        if rho_k > mu:
            xk += pk
            if rho_k >= mu_e:
                Delta_k = max(Delta_k, 2 * torch.norm(pk))
        else:
            Delta_k /= 2

        k += 1

    return xk, k


def setup_preconditioner(A): # ILU decomp for preconditioning
    A = A.double()
    indices = A.nonzero().t()
    values = A[indices[0], indices[1]]
    A_sparse = scipy.sparse.csc_matrix((values.cpu().numpy(), (indices[0].cpu().numpy(), indices[1].cpu().numpy())), shape=A.shape)

    M_inv = scipy.sparse.linalg.spilu(A_sparse, fill_factor=1)

    return lambda x: torch.tensor(M_inv.solve(x.cpu().numpy()), dtype=torch.float64, device=x.device)


def precond_conjugate_gradient(A, b, x0, max_iter=50, tol=1e-6):
    M_inv_op = setup_preconditioner(A)

    initial_r = b - torch.mv(A, x0)
    preconditioned_r = M_inv_op(initial_r)
    preconditioned_x0 = x0 + preconditioned_r

    final_x, num_iterations = conjugate_gradient(A, b, preconditioned_x0, max_iter, tol)

    return final_x, num_iterations


def evaluate_and_time(method, method_name, *args, **kwargs):
    start_time = time.time()
    x_min, iterations = method(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    relative_difference = torch.norm(x_min - x_true) / torch.max(torch.norm(x_min), torch.norm(x_true))
    
    print(f"{method_name}:")
    print(f"Minimizer: {x_min} after {iterations} iterations.")
    print(f"Relative Error: {relative_difference.item():.4e}")
    print(f"Time Elapsed: {elapsed_time:.5f} seconds\n")


evaluate_and_time(steepest_descent, "Steepest Descent Method", x0, max_iter=args.max_iter)
evaluate_and_time(newton_method, "Newton's Method", x0, max_iter=args.max_iter)
evaluate_and_time(newton_method_cg, "Newton + CG", x0, max_iter=args.max_iter)
evaluate_and_time(bfgs_quasi_newton, "BFGS Method", x0, max_iter=args.max_iter)
evaluate_and_time(trust_region, "Trust Region Method", x0, max_iter=args.max_iter, sub='direct')
evaluate_and_time(trust_region, "Trust Region + CG", x0, max_iter=args.max_iter, sub='cg')


if args.func == "quad":
    evaluate_and_time(conjugate_gradient, "Conjugate Gradient", A, b, x0, max_iter=args.max_iter)
    evaluate_and_time(precond_conjugate_gradient, "Preconditioned Conjugate Gradient", A, b, x0, max_iter=args.max_iter)