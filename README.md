# MATH 140: Mathematical Methods for Optimization

This repository contains all coursework and project materials for MATH 140: Mathematical Methods for Optimization, focusing on various optimization techniques such as Steepest Descent, Newton's Method, Quasi-Newton Methods, and Trust Region Methods.

## Homework Assignments

This repository includes completed homework assignments 1 through 7:

- [**HW2**: Norms and Properties](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW2/HW2.pdf) - Discusses various norms and their mathematical properties.
- [**HW3**: Convex Functions and Optimization](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW3/HW3.pdf) - Includes proofs and discussions on convex functions.
- [**HW4**: Gradient and Hessian Calculations](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW4/HW4.pdf) - Focuses on the calculation of gradients and Hessians for optimization.
- [**HW5**: Steepest Descent and Newton's Method](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW5/HW5.pdf) - Implements and compares Steepest Descent and Newton's Method with line search.
- [**HW6**: Quasi-Newton and SR1 Update](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW6/HW6.pdf) - Examines Quasi-Newton and properties of positive definite matrices and the SR1 update.
- [**HW7**: Trust Region Methods](https://github.com/napronald/MATH-140-Mathematical-Methods-for-Optimization/blob/main/HW7/HW7.pdf) - Detailed implementation and analysis of Trust Region Methods.

## Project Overview

The project folder includes implementations of all the optimization techniques covered in class, including enhancements using the Conjugate Gradient method to improve computational efficiency. Both the Rosenbrock function and a quadratic function are used to test these algorithms.

### Running the Project Script
To execute the project script, use the following command in your terminal:

```bash
python main.py --func quad --n_dim 2 --max_iter 100, --seed 0 --reg 1e9
```

Replace the parameters as needed or use the defaults provided in the Python script. Here's a breakdown of the command-line arguments:

- `--func`: Function to optimize (`quad` for quadratic, `rosen` for Rosenbrock).
- `--n_dim`: Dimensionality of the problem (e.g., 5000).
- `--seed`: Random seed for initial point generation.
- `--max_iter`: Maximum number of iterations.
- `--reg`: Regularization factor.

#### Prerequisites

Ensure the following software is installed on your system:
- Python 3.8 or newer
- scipy
- torch

Install the required packages using pip:

```bash
pip install numpy scipy torch
```

## Disclaimer
There are a few known issues and typos in the homework folders and potentially in the project folder. If you find any typos or have any suggestions, please feel free to contact me.
