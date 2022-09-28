import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import least_squares
from autograd import jacobian
import random

f_calc = np.zeros(3)


def exhaustive_search(func, bounds, eps):
    i = 0
    x_min = bounds[0]
    func_min = func(x_min)
    for x in np.arange(bounds[0], bounds[1], eps):
        i += 1
        if func(x) < func_min:
            func_min = func(x)
            x_min = x
    return x_min, func_min, i


def dichotomy_search(func, bounds, eps):
    i = 0
    delta = eps / 2
    a = bounds[0]
    b = bounds[1]
    while b - a >= eps:
        i += 1
        x_1 = (a + b - delta) / 2
        x_2 = (a + b + delta) / 2

        if func(x_1) <= func(x_2):
            b = x_2
        else:
            a = x_1

    x_min = np.around((a + b) / 2, decimals=5)
    func_min = func(x_min)
    return x_min, func_min, i


def golden_section_search(func, bounds, eps):
    i = 1
    a = bounds[0]
    b = bounds[1]

    x_1 = a + (3 - np.sqrt(5)) * (b - a) / 2
    x_2 = b + (-3 + np.sqrt(5)) * (b - a) / 2
    func_x1 = func(x_1)
    func_x2 = func(x_2)

    while b - a >= eps:
        if func_x1 <= func_x2:
            b = x_2
            x_2 = x_1
            x_1 = a + (3 - np.sqrt(5)) * (b - a) / 2
            func_x2 = func_x1
            func_x1 = func(x_1)
        else:
            a = x_1
            x_1 = x_2
            x_2 = b + (-3 + np.sqrt(5)) * (b - a) / 2
            func_x1 = func_x2
            func_x2 = func(x_2)
        i += 1

    x_min = np.around((a + b) / 2, decimals=5)
    func_min = func(x_min)
    return x_min, func_min, i


def linear_approximant(x, a, b):
    return a * x + b


def rational_approximant(x, a, b):
    return a / (1 + b * x)


def LSE_lin(*args):  # Least Squares Error
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def LSE_rat(*args):
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [rational_approximant(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def LSE_lin_lm(*args):  # Least Squares Error
    a = args[0][0]
    b = args[0][1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    f = np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])
    return [f, f]


def LSE_rat_lm(*args):
    a = args[0][0]
    b = args[0][1]
    xdata = [linear_approximant(x, a, b) for x in x_k]
    xdata = [rational_approximant(x, a, b) for x in x_k]
    f = np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])
    return [f, f]


def test_Exhaustive_search(func, N):
    a = np.linspace(0, 1, N + 1)
    b = np.linspace(0, 1, N + 1)
    a_best, b_best = None, None
    min_error = 10 ** 5
    f_calc = 0
    for i in range(len(a)):
        for j in range(len(b)):
            err = func(a[i], b[j])
            f_calc += 1
            if err < min_error:
                min_error = err
                a_best = a[i]
                b_best = b[j]
    return a_best, b_best, min_error, f_calc


def test_Gauss(func, a0, b0, N, eps):
    a, b = a0, b0
    a_grid = np.linspace(0, 1, N + 1)
    b_grid = np.linspace(0, 1, N + 1)
    min_f = func(a, b)
    step = 1
    f_calc = 1
    while step > eps / 2:
        a_best = a
        b_best = b
        for a_ in a_grid:
            err = func(a_, b)
            f_calc += 1
            if err < min_f:
                min_f = err
                a_best = a_
        step = abs(a - a_best)
        a = a_best
        for b_ in b_grid:
            err = func(a, b_)
            f_calc += 1
            if err < min_f:
                min_f = err
                b_best = b_
        step = min(step, abs(b - b_best))
        b = b_best
    return a, b, min_f, f_calc


def lab2_plot(func, approx, str, N, eps, alpha, beta, x_k, y_k):
    res1 = scipy.optimize.brute(func, ranges=(slice(0, 1, 1/(N+1)), (slice(0, 1, 1/(N+1)))))
    res2 = scipy.optimize.minimize(func, [0.5, 0.5], method='CG', tol=eps)
    res3 = scipy.optimize.minimize(func, [0.5, 0.5], method='Nelder-Mead', tol=eps)

    print(f'{str}\nExhaustive search:\na={round(res1[0], 5)}, b={round(res1[1], 5)}, '
          f'f={round(func(res1[0],res1[1]), 5)}\niterations={(N+1)**2}\n')
    print(f'Gauss:\na={round(res2.x[0], 5)}, b={round(res2.x[1], 5)}, '
          f'f={round(res2.fun, 5)}\niterations={res2.nfev}\n')
    print(f'Nelder-Mead:\na={round(res3.x[0],5)}, b={round(res3.x[1],5)}, '
          f'f={round(res3.fun,5)}\niterations={res3.nfev}\n')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [approx(x, res1[0], res1[1]) for x in x_k], label='Exhaustive search')
    plt.plot(x_k, [approx(x, res2.x[0], res2.x[1]) for x in x_k], label='Gauss')
    plt.plot(x_k, [approx(x, res3.x[0], res3.x[1]) for x in x_k], label='Nelder-Mead')
    plt.title(str)
    plt.legend()
    plt.savefig(str + ' lab2')
    plt.show()



def lab3_plot(func, func_lm, approx, str, N, eps, alpha, beta, x_k, y_k):
    res1 = scipy.optimize.minimize(func, x0=[0.5, 0.5], method='BFGS', tol=eps)
    res2 = scipy.optimize.minimize(func, x0=[0.5, 0.5], method='CG', tol=eps)
    res3 = scipy.optimize.minimize(func, x0=[0.5, 0.5], method='Newton-CG', jac=jacobian(func), tol=eps)
    # res44 = scipy.optimize.curve_fit(approx, xdata=x_k, ydata=y_k, method='lm')
    res4 = least_squares(func_lm, x0=[0.5, 0.5],  method='lm', gtol=eps, max_nfev=int(1e6))
    print(f'{str}\nGradient descent:\na={round(res1.x[0], 7)}, b={round(res1.x[1], 7)}, '
          f'f={round(res1.fun,7)}\niterations={res1.nfev}\n')
    print(f'Conjugate gradient descent:\na={round(res2.x[0], 7)}, b={round(res2.x[1], 7)}, '
          f'f={round(res2.fun,7)}\niterations={res2.nfev}\n')
    print(f'Newton:\na={round(res3.x[0], 7)}, b={round(res3.x[1], 7)}, '
          f'f={round(res3.fun,7)}\niterations={res3.nfev}\n')
    print(f'Levenberg-Marquardt:\na={round(res4.x[0], 7)}, b={round(res4.x[1], 7)}, '
          f'f={round(res4.fun[0],7)}\niterations={res4.nfev}\n')

    plt.plot(x_k, y_k, 'o')
    plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
    plt.plot(x_k, [approx(x, res1.x[0], res1.x[1]) for x in x_k], label='Gradient descent')
    plt.plot(x_k, [approx(x, res2.x[0], res2.x[1]) for x in x_k], label='Conjugate gradient descent')
    plt.plot(x_k, [approx(x, res3.x[0], res3.x[1]) for x in x_k], label='Newton')
    plt.plot(x_k, [approx(x, res4.x[0], res4.x[1]) for x in x_k], label='Levenberg-Marquardt')
    # plt.plot(x_k, [approx(x, res44[0][0], res44[0][1]) for x in x_k], label='Levenberg-Marquardt v1')
    plt.title(str)
    plt.legend()
    plt.savefig(str + ' lab3')
    plt.show()


N = 100
eps = 0.001
alpha = random.random()
beta = random.random()
noise = np.random.normal(0, 1, N + 1)
x_k = np.array([k / N for k in range(N + 1)])
y_k = np.array([alpha * x_k[k] + beta + noise[k] for k in range(len(x_k))])


lab2_plot(LSE_lin, linear_approximant, 'Linear approximation', N, eps, alpha, beta, x_k, y_k)
lab2_plot(LSE_rat, rational_approximant, 'Rational approximation', N, eps, alpha, beta, x_k, y_k)

lab3_plot(LSE_lin, LSE_lin_lm, linear_approximant, 'Linear approximation', N, eps, alpha, beta, x_k, y_k)
lab3_plot(LSE_rat, LSE_rat_lm, rational_approximant, 'Rational approximation', N, eps, alpha, beta, x_k, y_k)