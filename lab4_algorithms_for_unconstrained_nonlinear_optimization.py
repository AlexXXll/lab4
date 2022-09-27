import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random
from abc import ABCMeta, abstractmethod


class Vector(object):
    def __init__(self, x, y):
        """ Create a vector, example: v = Vector(1,2) """
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Vector(x, y)

    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Vector(x, y)

    def c(self):
        return (self.x, self.y)


# objective function
def f(point):
    x, y = point
    return x ** 2 + x * y + y ** 2 - 6 * x - 9 * y
def nelder_mead(alpha=1, beta=0.5, gamma=2, maxiter=10):
    # initialization
    v1 = Vector(0, 0)
    v2 = Vector(1.0, 0)
    v3 = Vector(0, 1)

    for i in range(maxiter):
        adict = {v1: f(v1.c()), v2: f(v2.c()), v3: f(v3.c())}
        points = sorted(adict.items(), key=lambda x: x[1])

        b = points[0][0]
        g = points[1][0]
        w = points[2][0]

        mid = (g + b) / 2

        # reflection
        xr = mid + alpha * (mid - w)
        if f(xr.c()) < f(g.c()):
            w = xr
        else:
            if f(xr.c()) < f(w.c()):
                w = xr
            c = (w + mid) / 2
            if f(c.c()) < f(w.c()):
                w = c
        if f(xr.c()) < f(b.c()):

            # expansion
            xe = mid + gamma * (xr - mid)
            if f(xe.c()) < f(xr.c()):
                w = xe
            else:
                w = xr
        if f(xr.c()) > f(g.c()):

            # contraction
            xc = mid + beta * (w - mid)
            if f(xc.c()) < f(w.c()):
                w = xc

        # update points
        v1 = w
        v2 = g
        v3 = b
    return b

class Optimizer:
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, epsilon=1e-7, function_array=None, metaclass=ABCMeta):
        self.function_array = function_array
        self.epsilon = epsilon
        self.interval = interval
        self.function = function
        self.gradient = gradient
        self.hesse = hesse
        self.jacobi = jacobi
        self.name = self.__class__.__name__.replace('Optimizer', '')
        self.x = initialPoint
        self.y = self.function(initialPoint)

    "This method will return the next point in the optimization process"
    @abstractmethod
    def next_point(self):
        pass

    """
    Moving to the next point.
    Saves in Optimizer class next coordinates
    """

    def move_next(self, nextX):
        nextY = self.function(nextX)
        self.y = nextY
        self.x = nextX
        return self.x, self.y

class LevenbergMarquardtOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, function_array=None, learningRate=1):
        self.learningRate = learningRate
        functionNew = lambda x: np.array([function(x)])
        super().__init__(functionNew, initialPoint, gradient, jacobi, hessian, interval, function_array=function_array)
        self.v = 2
        self.alpha = 1e-3
        self.m = self.alpha * np.max(self.getA(jacobi(initialPoint)))

    def getA(self, jacobi):
       return np.dot(jacobi.T, jacobi)

    def getF(self, d):
       function = self.function_array(d)
       return 0.5 * np.dot(function.T, function)

    def next_point(self):
        if self.y==0: # finished. Y can't be less than zero
            return self.x, self.y
        jacobi = self.jacobi(self.x)
        A = self.getA(jacobi)
        g = np.dot(jacobi.T, self.function_array(self.x)).reshape((-1, 1))
        leftPartInverse = np.linalg.inv(A + self.m * np.eye(A.shape[0], A.shape[1]))
        d_lm = - np.dot(leftPartInverse, g) # moving direction
        x_new = self.x + self.learningRate * d_lm.reshape((-1)) # line search
        grain_numerator = (self.getF(self.x) - self.getF(x_new))
        gain_divisor = 0.5* np.dot(d_lm.T, self.m*d_lm-g) + 1e-10
        gain = grain_numerator / gain_divisor
        if gain > 0: # it's a good function approximation.
            self.move_next(x_new) # ok, step acceptable
            self.m = self.m * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            self.v = 2
        else:
            self.m *= self.v
            self.v *= 2

        return self.x, self.y

    def lab22_plot(N, x_k, y_k):
        res1_lin = scipy.optimize.brute(LSE_lin, ranges=(slice(0, 1, 1 / (N + 1)), (slice(0, 1, 1 / (N + 1)))))
        res2_lin = scipy.optimize.minimize(LSE_lin, [0.5, 0.5], method='CG', options={'eps': eps})
        res3_lin = scipy.optimize.minimize(LSE_lin, [0.5, 0.5], method='Nelder-Mead')

        plt.plot(x_k, y_k, 'o')
        plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
        plt.plot(x_k, [linear_approximant(x, res1_lin[0], res1_lin[1]) for x in x_k], label='Exhaustive search')
        plt.plot(x_k, [linear_approximant(x, res2_lin.x[0], res2_lin.x[1]) for x in x_k], label='Gauss')
        plt.plot(x_k, [linear_approximant(x, res3_lin.x[0], res3_lin.x[1]) for x in x_k], label='Nelder-Mead')
        plt.title('Linear approximation')
        plt.legend()
        plt.savefig('Linear approximation')
        plt.show()

        res1_rat = scipy.optimize.brute(LSE_rat, ranges=(slice(0, 1, 1 / (N + 1)), (slice(0, 1, 1 / (N + 1)))))
        res2_rat = scipy.optimize.minimize(LSE_rat, [0.5, 0.5], method='CG', options={'eps': eps})
        res3_rat = scipy.optimize.minimize(LSE_rat, [0.5, 0.5], method='Nelder-Mead')

        plt.plot(x_k, y_k, 'o')
        plt.plot(x_k, [linear_approximant(x, alpha, beta) for x in x_k], label='Generating line')
        plt.plot(x_k, [rational_approximant(x, res1_rat[0], res1_rat[1]) for x in x_k], label='Exhaustive search')
        plt.plot(x_k, [rational_approximant(x, res2_rat.x[0], res2_rat.x[1]) for x in x_k], label='Gauss')
        plt.plot(x_k, [rational_approximant(x, res3_rat.x[0], res3_rat.x[1]) for x in x_k], label='Nelder-Mead')
        plt.title('Rational approximation')
        plt.legend()
        plt.savefig('Rational approximation')
        plt.show()

    N = 1000
    eps = 0.001
    noise = np.random.normal(0, 1, N + 1)
    x_k = np.array([3 * k / N for k in range(N + 1)])
    y_k = np.array([alpha * x_k[k] + beta + noise[k] for k in range(len(x_k))])