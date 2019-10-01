import cython
import math
from scipy.optimize import brent
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cpdef inverted_MPeB(np.ndarray X,float c,float mu_):
	'''Returns confidence level given a lower bound

	Parameters
	----------
	X : Vector of n integers.
		Independently distributed random variables
	n : int
		Length of vectors
	c : int.
		Threshold. Must be empirically fixed
	mu_ : int
		Lower bound.

	Returns
	-------
	float
		Confidence level for the threshold
	'''
	cdef int n = X.shape[0]
	cdef np.ndarray Y = np.zeros([n], dtype=DTYPE)
	cdef double[:] Y_view = Y
	cdef double[:] X_view = X
	#cdef np.ndarray Y2 = np.zeros([n], dtype=DTYPE)
	#cdef np.ndarray Y3 = np.zeros([n], dtype=DTYPE)
	cdef float k1,k2,k2_1,k3,z_factor1,z
	cdef int i
	cdef float Y2 = 0
	cdef float Y3 = 0

	for i in range(n):
		Y_view[i] = min(X_view[i],c)
		Y2 = Y2 + (Y_view[i]/c)
		Y3 = Y3 + ((Y_view[i]/c)**2)

	k1 = (7*n) / (3*(n-1))
	k3 = mu_ * (1/c) * n - Y2
	k2_1 = n * Y3 - Y2 ** 2
	k2 = sqrt( (2 * k2_1) / (n-1) )
	z_factor1 = k2**2 - 4*k1*k3
	if z_factor1 > 0:
		z = (-k2 + sqrt(z_factor1)) / (2*k1)
		if z > 0:
			return 1-min(1,2*exp(-(z**2)))
		else:
			return 0
	else:
		return 0

cpdef lower_bound_calc(float c,np.ndarray X,float delta):
	'''Returns likelihood of threshold c value

	Parameters
	----------
	c : int.
		Threshold. Must be empirically fixed
	X : Vector of n integers.
		Independently distributed random variables
	n : int
		Length of X vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Likelihood of c threshold
	'''
	cdef int n = X.shape[0]
	cdef np.ndarray Y = np.zeros([n], dtype=DTYPE)
	cdef double[:] Y_view = Y
	cdef double[:] X_view = X
	#cdef np.ndarray Y2 = np.zeros([n], dtype=DTYPE)
	#cdef np.ndarray Y3 = np.zeros([n], dtype=DTYPE)
	cdef float f1,f2,f3,f4_1,f4_2,f4,f5
	cdef int i
	cdef float Y2 = 0
	cdef float Y3 = 0

	for i in range(n):
		Y_view[i] = min(X_view[i],c)
		Y2 = Y2 + (Y_view[i]/c)
		Y3 = Y3 + ((Y_view[i]/c)**2)

	f1 = (n/c)**(-1)
	f2 = Y2
	f3 = (7*n*log(2/delta)) / (3*(n - 1))
	f4_1 = (2*log(2/delta)) / (n-1)
	f4_2 = n*Y3-Y2**2
	f4 = sqrt(f4_1*f4_2)
	f5 = f2-f3-f4
	return f1*f5

cpdef c_like(float c,np.ndarray X_pre,int n_post,float delta):
	'''Returns likelihood of threshold c value

	Parameters
	----------
	c : int.
		Threshold. Must be empirically fixed
	X_pre : Vector of n_pre integers.
		Independently distributed random variables
	n_pre : int
		Length of X_pre vector
	n_post : int
		Length of X_post vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Likelihood of c threshold
	'''
	cdef int n_pre = X_pre.shape[0]
	cdef np.ndarray Y = np.zeros([n_pre], dtype=DTYPE)
	cdef double[:] Y_view = Y
	cdef double[:] X_pre_view = X_pre
	cdef float f1,f2_1,f2_2,f2,f3_1,f3_2,f3_3,f3
	cdef int i
	cdef float Y1 = 0
	cdef float Y2 = 0

	for i in range(n_pre):
		Y_view[i] = min(X_pre_view[i],c)
		Y1 = Y1 + Y_view[i]
		Y2 = Y2 + (Y_view[i]**2)

	f1 = Y1/n_pre
	f2_1 = 7*c*log(2/delta)
	f2_2 = 3*(n_post - 1)
	f2 = f2_1 / f2_2
	f3_1 = log(2/delta) / n_post
	f3_2 = 2 / (n_pre*(n_pre-1))
	f3_3 = n_pre * Y2 - Y1**2
	f3 = sqrt(f3_1*f3_2*f3_3)
	return -(f1-f2-f3)


cpdef optimize_threshold(np.ndarray X_pre,int n_post,float delta):
	'''Returns likelihood of threshold c value

	Parameters
	----------
	X_pre : Vector of n_pre integers.
		Independently distributed random variables
	n_pre : int
		Length of X_pre vector
	n_post : int
		Length of X_post vector
	delta : int
		1 - Confidence level for the threshold

	Returns
	-------
	float
		Optimal c threhold
	'''
	cdef int n_pre = X_pre.shape[0]
	cdef float c_optim

	c_optim, fval, iter_, funcalls = brent(c_like,(X_pre,n_post,delta),brack=(0,1000),full_output=True)
	return c_optim

def generate_bounds(np.ndarray X,float delta):
	#Divide dataset for optimization
	cdef int N = X.shape[0]
	cdef int mean = round(np.mean(X))
	cdef int n_pre = round(N*0.05)
	cdef int n_post = N-n_pre
	cdef np.ndarray X_pre, X_post
	cdef float c_opt, p, m, lower_bound
	cdef int i
	cdef np.ndarray confidences = np.zeros([mean*2-2], dtype=DTYPE)
	cdef np.ndarray bounds = np.zeros([mean*2-2], dtype=DTYPE)
	
	np.random.shuffle(X)
	X_pre, X_post = X[:n_pre], X[n_pre:]
	#optimize threshold
	c_opt = optimize_threshold(X_pre,n_post,delta)
	#bounds
	lower_bound = lower_bound_calc(c_opt,X_post,0.9)
	#calculate bounds
	i = 0
	for m in np.arange(0,mean-1,0.5):
		bounds[i] = m
		confidences[i] = inverted_MPeB(X_post,c_opt,m)
		i = i+1
	return X_post, lower_bound, bounds, confidences