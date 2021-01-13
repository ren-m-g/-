import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sb
from scipy.special import factorial
from scipy.stats import multivariate_normal, spearmanr
from typing import *
import numpy as np
from math import sqrt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import minimize


sample10 = 10
sample50 = 50
sample1000 = 1000
sample100 = 100
sample20 = 20
sample60 = 60

#lab1
def norm(sample):
    mu = 0
    sigma = 1
    s = np.random.normal(mu, sigma, sample)
    plt.hist(s, bins=10,density=True)

    x = np.linspace(min(s), max(s), 1000)
    y = stats.norm.pdf(x, loc=0, scale=math.sqrt(1))
    plt.plot(x, y)
    plt.show()

def cauchy(sample):
    mu = 0
    lamda = 1
    s = stats.cauchy.rvs(size=sample)
    plt.hist(s, bins=10, density=True)

    x = np.linspace(min(s), max(s), 1000)
    y = stats.cauchy.pdf(x, loc=mu, scale=lamda)
    plt.plot(x, y)
    plt.show()

def laplace(sample):
    mu = 0
    b = 1 / math.sqrt(2)
    s = np.random.laplace(mu, b, sample)
    plt.hist(s, bins=10, density=True)

    x = np.linspace(min(s), max(s), 1000)
    y = stats.laplace.pdf(x, loc=mu, scale=b)
    plt.plot(x, y)
    plt.show()

def poisson(sample):
    k = 10
    s = np.random.poisson(k, sample)
    plt.hist(s, bins=20, density=True)

    x = np.arange(20)
    y = stats.poisson.pmf(x, mu=k)
    plt.plot(x, y)
    plt.show()

def uniform(sample):
    a = - math.sqrt(3)
    b = math.sqrt(3)
    s = np.random.uniform((- math.sqrt(3)), math.sqrt(3), sample)
    plt.hist(s, bins=10, density=True)

    x = np.linspace(min(s), max(s), 1000)
    y = stats.uniform.pdf(x, loc=a, scale=b - a)
    plt.plot(x, y)
    plt.show()

#lab2
def suansu(distr, sample):
    sample1 = []
    sample2 = []
    sample3 = []
    sample4 = []
    sample5 = []
    for i in range(1000):
        if distr == 1:
            s = np.random.normal(0, 1, sample)

        if distr == 2:
            s = np.random.standard_cauchy(sample)

        if distr == 3:
            s = np.random.laplace(0, 1 / math.sqrt(2), sample)

        if distr == 4:
            s = np.random.poisson(10, sample)

        if distr == 5:
            s = np.random.uniform((- math.sqrt(3)), math.sqrt(3), sample)

        s.sort()
        x = np.mean(s)
        sample1.append(x)
        med_x= np.median(s)
        sample2.append(med_x)
        z_R=(s.max() + s.min())/2
        sample3.append(z_R)
        z_Q = (np.quantile(s, 0.25, interpolation='lower') + np.quantile(s, 0.75, interpolation='higher')) / 2
        sample4.append(z_Q)
        delete_len = len(s) / 4
        s = s[int(delete_len): int(-delete_len)]
        z_tr=np.mean(s)
        sample5.append(z_tr)
    e_x = np.mean(sample1)
    e_med_x = np.mean(sample2)
    e_z_R = np.mean(sample3)
    e_z_Q = np.mean(sample4)
    e_z_tr = np.mean(sample5)
    d_x = np.var(sample1)
    d_med_x = np.var(sample2)
    d_z_R = np.var(sample3)
    d_z_Q = np.var(sample4)
    d_z_tr = np.var(sample5)
    print("e_x=", e_x, " ", "e_med_x=", e_med_x, " ", "e_z_R=", e_z_R, " ", "e_z_Q=", e_z_Q, " ", "e_z_tr=", e_z_tr)
    print("d_x=", d_x, " ", "d_med_x=", d_med_x, " ", "d_z_R=", d_z_R, " ", "d_z_Q=", d_z_Q, " ", "d_z_tr=", d_z_tr)

#suansu(1,sample10)


#lab3
def b_norm():
    s20 = np.random.normal(0, 1, 20)
    s1000 = np.random.normal(0, 1, 100)
    tit = 'Normal'
    return (s20,s1000,tit)

def b_cauchy():
    s20 = np.random.standard_cauchy(20)
    s1000 = np.random.standard_cauchy(100)
    tit = 'Cauchy'
    return (s20, s1000, tit)

def b_laplace():
    s20 = np.random.laplace(0, 1 / (math.sqrt(2)), 20)
    s1000 = np.random.laplace(0, 1 / (math.sqrt(2)), 100)
    tit = 'Laplace'
    return (s20, s1000, tit)

def b_poisson():
    s20 = np.random.poisson(10, 20)
    s1000 = np.random.poisson(10, 100)
    tit = 'Poisson'
    return (s20, s1000, tit)

def b_uniform():
    s20 = np.random.uniform((- math.sqrt(3)), math.sqrt(3), 20)
    s1000 = np.random.uniform((- math.sqrt(3)), math.sqrt(3), 100)
    tit = 'Uniform'
    return (s20, s1000, tit)

def boxplot(b):
    lis=b()
    labelss = 'n = 20', 'n = 100'
    plt.boxplot([lis[0], lis[1]], vert=False, labels=labelss)
    plt.title(lis[2])
    plt.show()
#boxplot(b_uniform)

def get_perc(num,sample):
    if num == 1:
        s = np.random.normal(0, 1, sample)
    if num == 2:
        s = np.random.standard_cauchy(sample)
    if num == 3:
        s = np.random.laplace(0, 1 / (math.sqrt(2)), sample)
    if num == 4:
        s = np.random.poisson(10, sample)
    if num == 5:
        s = np.random.uniform((- math.sqrt(3)), math.sqrt(3), sample)
    s.sort()
    q1 = np.quantile(s, 0.25, interpolation='lower')
    q3 = np.quantile(s, 0.75, interpolation='higher')
    x1 = q1 - (3 / 2) * (q3 - q1)
    x2 = q3 + (3 / 2) * (q3 - q1)
    count = 0
    for r in s:
        if r > x2 or r < x1:
            count += 1
    print(count / len(s))


#lab4
def ecdf(distr, sample):
    title = ''
    if distr == 1:
        s = np.random.normal(0, 1, sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.cdf(x, 0, 1)
        title = 'Normal n=%d' % sample
    elif distr == 2:
        s = np.random.standard_cauchy(sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.cauchy.cdf(x)
        title = 'Cauchy n=%d' % sample
    elif distr == 3:
        s = np.random.laplace(0, 1 / (math.sqrt(2)), sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.laplace.cdf(x, 0, 1 / (math.sqrt(2)))
        title = 'Laplace n=%d' % sample
    elif distr == 4:
        s = np.random.poisson(10, sample)
        x = np.linspace(6, 14, 1000)
        y = stats.poisson.cdf(x, 10)
        title = 'Poisson n=%d' % sample
    elif distr == 5:
        s = np.random.uniform((- math.sqrt(3)), math.sqrt(3), sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.uniform.cdf(x, (- math.sqrt(3)), 2 * math.sqrt(3))
        title = 'Uniform n=%d' % sample
    ecdf = ECDF(s)
    if distr == 4:
        ecdf.x[0] = 6
        ecdf.x[len(ecdf.x) - 1] = 14
        plt.axis([6, 14, 0, 1.1])
    else:
        ecdf.x[0] = -4
        ecdf.x[len(ecdf.x) - 1] = 4
        plt.axis([-4, 4, 0, 1.1])
    plt.plot(x, y, color='red')
    plt.step(ecdf.x, ecdf.y)
    plt.title(title)
    plt.show()

#ecdf(5,100)

def kde(distr, sample, hue):
    if distr == 4:
        r = (6, 14)
    else:
        r = (-4, 4)
    if distr == 1:
        s = np.random.normal(0, 1, sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        tit = 'Normal n=%d' % sample
        plt.axis([-4, 4, 0.0, 1.0])
    elif distr == 2:
        s = np.random.standard_cauchy(sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.cauchy.pdf(x, 0, 1)
        tit = 'Cauchy n=%d' % sample
        plt.axis([-4, 4, 0.0, 1.0])
    elif distr == 3:
        s = np.random.laplace(0, 1 / (math.sqrt(2)), sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.laplace.pdf(x, 0, 1 / (math.sqrt(2)))
        tit = 'Laplace n=%d' % sample
        plt.axis([-4, 4, 0.0, 1.0])
    elif distr == 4:
        s = np.random.poisson(10, sample)
        x = np.linspace(6, 14, 1000)
        y = np.exp(-10) * np.power(10, x) / factorial(x)
        tit = 'Poisson n=%d' % sample
        plt.axis([6, 14, 0.0, 1.0])
    elif distr == 5:
        s = np.random.uniform((- math.sqrt(3)), math.sqrt(3), sample)
        x = np.linspace(-4, 4, 1000)
        y = stats.uniform.pdf(x, - math.sqrt(3), 2 * math.sqrt(3))
        tit = 'Uniform n=%d' % sample
        plt.axis([-4, 4, 0.0, 1.0])
    s = s[s <= r[1]]
    s = s[s >= r[0]]
    kde = stats.gaussian_kde(s)
    kde.set_bandwidth(bw_method='silverman')
    hn = kde.factor
    if hue == 1:
        sb.kdeplot(s, bw=(hn / 2), palette='Blues_r')
        plt.plot(x, y, color='red')
    elif hue == 2:
        sb.kdeplot(s, bw=hn, palette='Blues_r')
        plt.plot(x, y, color='red')
    elif hue == 3:
        sb.kdeplot(s, bw=2 * hn, palette='Blues_r')
        plt.plot(x, y, color='red')
    plt.title(tit)
    plt.show()


