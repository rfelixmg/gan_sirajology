from sympy.functions.elementary.exponential import log

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import pylab
from scipy.stats import norm


#Logaritm function
from numpy.linalg.linalg import norm


def log():
    n_range = 10

    x = np.array(np.arange(n_range).astype(np.float)) + 1
    y = norm.pdf( x)
    #y = np.log2(x/2)

    plt.plot(x, y)
    plt.ylabel('exp2(x) = 2^x')
    plt.xlabel('x')

    plt.show()

def gaussian_distribution(xx):
    mu = np.mean(xx)
    sigma = np.std(xx)

    y = []
    for x in xx:
        yy = (1./(np.sqrt( 2 * np.pi) * sigma)) * np.exp(-0.5 * (((x-mu)**2)/(sigma**2)))
        y.append(yy)
    y = np.array(y)

    plt.bar( xx, y, 0.0001, color='blue')
    plt.ylabel('Distribution p(x)')
    plt.xlabel('Distribution of X | mu:%f, sig:%f'%( mu, sigma))

    plt.show()


#Binomial Distribution
def bin_distribution(n, p):
    # n = 10.0
    # p = 0.5

    rect = 0, 0, (n + 10), p

    mu = n * p
    sigma = (n*p)*(1-p)
    width = 0.35



    x = np.array(np.arange(n + 1).astype(np.float))

    y = []
    for xx in x:
        yy = (math.factorial(n)/(math.factorial(xx) * math.factorial((n-xx)))) * (p**xx) * ((1 - p)**(n-xx))
        y.append(yy)
    y = np.array(y)


    plt.bar(x,y, 0.35, color='blue')

    plt.ylabel('Distribution for %d temptations' % n)
    plt.xlabel('x successful temptations| mu:%f, sig:%f'%(mu,sigma))

    plt.show()


if __name__ == "__main__":

    #log()
    #bin_distribution(40, 0.8)
    x = np.random.uniform(-10, 10, size=10000)
    #x = np.array(np.arange(100).astype(np.float)) + 1
    gaussian_distribution(x)
