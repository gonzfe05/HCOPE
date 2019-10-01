import numpy as np
from mpeb_python import generate_bounds as gb
from mpeb_c import generate_bounds as gb_c
import matplotlib.pyplot as plt
from time import perf_counter


#params
scale = 100
N = 1000
X = np.random.exponential(scale,N)
delta = 0.9
t1 = perf_counter()
X_post, lower_bounds, bounds, confidences = gb(X,delta)
t2 = perf_counter()
print("Elapsed seconds in python:", t2-t1) 
t1 = perf_counter()
X_post, lower_bound, bounds, confidences = gb_c(X,delta)
t2 = perf_counter()
print("Elapsed seconds in cython:", t2-t1) 
#prepare hist of bounds
hist, bin_edges = np.histogram(X_post, bins = scale+1, density=True)
for i in range(1,len(hist)):
	hist[i] += hist[i-1]

#plot
fig, ax = plt.subplots()
bin_edges = [v for v in bin_edges if v <= scale]
hist = hist[:len(bin_edges)]
plt.plot(bin_edges,hist, label='Truncated empirical distribution')
plt.plot(bounds,confidences,'--', label='MPeB inequality confidence bound')
plt.axvline(x=lower_bound,ls=':')
ax.legend()
ax.grid(True)
plt.xlabel('Values') 
plt.ylabel('Probability') 
plt.yticks(np.arange(0, 1.1, 0.1)) 
_ = plt.show()