import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 100, 10000

def deriv_lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    x, y, z = X
    dx_dt = -sigma*(x - y)
    dy_dt = rho*x - y - x*z
    dz_dt = -beta*z + x*y
    return dx_dt, dy_dt, dz_dt

# Integrate the Lorenz equations on the time grid t
t = np.linspace(0, tmax, n)
f = odeint(deriv_lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

# Make the line multi-coloured by plotting it in segments of length s which
# change in colour across the whole time series.
s = 10
c = np.linspace(0,1,n)
for i in range(0,n-s,s):
    ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1,c[i], .2*c[i]), alpha=0.8, lw=.4)

# Remove all the axis clutter, leaving just the curve.
ax.set_axis_off()

plt.show()
