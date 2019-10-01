# HCOPE: High Confidence Off-Policy Evaluation

## Tool for finding the lower bound on a distribution mean with a fixed confidence value

Based on Thomas et al. High Confidence Off-Policy Evaluation. It uses concentration inequality to get a confidence bound on the expectation of a random variable, with the only assumption being independently sampled values. We use Maurer & Pontil’s empirical Bernstein (MPeB) inequality which replaces the true (unknown) variance in Bernstein’s inequality with the sample variance. The *c* threshold must be fitted to the data, so we find the optimal on a 5% sample and compute the bounds on the other 95% of the data.

## Structure

This repo was implemented as a test to compare the gains in speed when shifting from python (mpeb_python.py) to cython (mpeb_c.pyx), which resulted in a **35x** faster run.
The file test_bounds.py will run both implementations on a simulated exponential, time the difference and plot the result from the cython implementation.
Regarding the implementations, they can be used with, they can be used with a call to **generate_bounds**, it takes in the sample and a confidence level, and will output the *X_post* (sample used to compute the bounds), *lower_bound* (90% confidence x), *bounds* (list of x values), *confidences* (confidence level of each x in bounds).

## Compiling

The cython code can be compiled with: *python setup.py build_ext --inplace*
