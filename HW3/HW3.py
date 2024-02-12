import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

print("\n")
print(f'1.0 Double Precision:{np.spacing(np.float64(1))}')
print("\n")
print(f'1.0 Single Precision:{np.spacing(np.float32(1))}')
print("\n")
print(f'2^40 Double Precision:{np.spacing(np.float64((2**40)))}')
print("\n")
print(f'2^40 Single Precision:{np.spacing(np.float32((2**40)))}')
print("\n")

x = sp.symbols('x')
f = sp.tan(x)
a = sp.pi/4

p1 = 1 + 2*(x) - np.pi/2
p2 = 1 + 2*(x - a) + 2*((x - a)**2)

f_func = sp.lambdify(x, f, 'numpy')
p1_func = sp.lambdify(x, p1, 'numpy')
p2_func = sp.lambdify(x, p2, 'numpy')

x = np.linspace(0, np.pi/4, 100)
fx = f_func(x)
p1x = p1_func(x)
p2x = p2_vals = p2_func(x)

plt.figure(figsize=(12, 8))
plt.plot(x, fx, label='f(x) = tan(x)')
plt.plot(x, p1x, '-', label='p1(x)')
plt.plot(x, p2x, '-', label='p2(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Taylor Polynomials')
plt.grid(True)
plt.show()

r1 = fx - p1x
r2 = fx - p2x

plt.figure(figsize=(12, 8))
plt.plot(x, r1, '--', label='r1(x)')
plt.plot(x, np.abs(r2), ':', label='r2(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('r(x)')
plt.title('Reminders')
plt.grid(True)
plt.show()

b1 = 2*(x - a)**2
b2 = (12.83/6)*(x - a)**3

abs_r1 = np.abs(r1)
abs_r2 = np.abs(r2)

b_r1 = np.all(abs_r1 <= b1)
b_r2 = np.all(abs_r2 <= np.abs(b2))

print(b_r1)
print(b_r2)

plt.figure(figsize=(12, 8))
plt.plot(x, r1, '--', label='r1(x)')
plt.plot(x, np.abs(r2), ':', label='r2(x)')
plt.plot(x, b1, label='Bound for r1(x)')
plt.plot(x, np.abs(b2), label='Bound for r2(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('r(x)')
plt.title('Reminders and Theoretical Bounds')
plt.grid(True)
plt.show()

x = 1.2
h = np.zeros(60)
Dhf = np.zeros(60)
error = np.zeros(60)

for k in range(60):
    h[k] = 1 / (2 ** k)
    Dhf[k] = (np.sin(x + h[k]) - np.sin(x)) / h[k]
    error[k] = abs(np.cos(x) - Dhf[k])

print('h          f\'(x)               f\'(x) app          error     ratio')
for k in range(60):
    if k == 0:
        print(f'{h[k]:5.3e} {np.cos(x):19.16f} {Dhf[k]:19.16f} {error[k]:6.3e} N/A')
    else:
        ratio = error[k-1] / error[k]
        print(f'{h[k]:5.3e} {np.cos(x):19.16f} {Dhf[k]:19.16f} {error[k]:6.3e} {ratio:4.2f}')

plt.figure(figsize=(10, 6))
plt.loglog(h, error, label='Numerical Error')
plt.loglog(h, h/2, label='Theoretical Error (h/2)')
plt.loglog(h, np.finfo(float).eps / h, label='Machine Epsilon / h')
plt.xlim([h[-1], h[0]])
plt.xlabel('h')
plt.ylabel('Absolute Error')
plt.legend()
plt.title('Error in Numerical Derivative of sin(x)')
plt.show()

def f(x):
    return np.cos(x) - x

def df(x):
    return -np.sin(x) - 1

def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    results = [x0, x1]
    for n in range(2, max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        x2 = x1 - fx1*(x1 - x0)/(fx1 - fx0)
        results.append(x2)
        if abs(f(x2)) < tol:
            return results
        x0, x1 = x1, x2
    return results

def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    results = [x0]
    xn = x0
    for n in range(1, max_iter):
        fxn = f(xn)
        dfxn = df(xn)
        xn = xn - fxn/dfxn
        results.append(xn)
        if abs(fxn) < tol:
            return results
    return results

x0_fp = 0.5
x1_fp = np.pi/4

x0_secant = 0.5
x1_secant = np.pi/4
x0_newton = np.pi/4

secant_results = secant_method(f, x0_secant, x1_secant)
newton_results = newton_method(f, df, x0_newton)

print(f"{'n':<3} {'Secant':<20} {'Newton':<20}")
max_iter = max(len(secant_results), len(newton_results))  
for n in range(max_iter):
    secant = f"{secant_results[n]:<20.10f}" if n < len(secant_results) else ' ' * 20
    newton = f"{newton_results[n]:<20.10f}" if n < len(newton_results) else ' ' * 20
    print(f"{n:<3} {secant} {newton}")