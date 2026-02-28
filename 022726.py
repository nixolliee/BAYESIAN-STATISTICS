# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 09:58:05 2026

@author: ACER
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1
# using the uniform distribution for clarity  but we can also make the beta distribution look 
# completely ft by tweaking alpha and beta

uniform_dist = uniform_dist/uniform_dist.sum(0)
# normalizing the distribution to make the probabilities densities sum into 1
beta_dist = sts.beta.pdf(mu, 2, 5, loc = 1.65, scale = 0.2)
beta_dist = beta_dist/beta_dist.sum(0)
plt.plot(mu, beta_dist, label = 'Beta Dist')
plt.plot(mu, uniform_dist, label = 'Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability Density")
plt.legend()

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    # mu here is an array of values so the input is also an array
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()

import scipy as sp
# with warning because my code has imported the library scipy with the "sp" (import scipy as sp), 
# but i haven't actually used sp anywhere in my code afterward

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()

#notes
#scipy = big library for science and math in python
#scipy.stats = part of scipy that focuses on statistics
#numpy = main tool for working with numbers and arrays
#matplotlib = library to make graphs and charts
#pyplot = part of matplotlib that creates plots
#import = keyword that tells python to bring in an external module or library so i can use its features
#as = give the module or library a shorter nickname in my code
#linspace = creates a list of numbers between start and stop)
#sts.uniform.pdf = gives probability density of a uniform distribution at the points x
#sts.beta.pdf = gives the probability density at x for a beta distribution with paramenters alpha and beta (modeling priot beliefs)
#beta_dist.sum = summing up the probability densities in an array (to normalize a discrete approximation of a continuous distribution) 
#likelihood_func = calculates the likelihood of observing the data given a parameter
#datum = just a single piece of data