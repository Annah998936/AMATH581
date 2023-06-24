import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import csv

## Problem 1
dydt = lambda t, y: -3*y*np.sin(t)
y0 = np.pi/np.sqrt(2)
ytrue = lambda t: np.pi*np.exp(3*(np.cos(t) - 1))/np.sqrt(2)

# a - Forward Euler
def forward_euler(f, t, y0):
	dt = t[2] - t[1]
	y = np.zeros(len(t))
	y[0] = y0
	for k in range(len(y)-1):
		y[k+1] = y[k] + dt*f(t[k], y[k])

	return y

dt_vals = 2**(-np.linspace(2, 8, 7))
err = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	N = int(5/dt);
	t = np.linspace(0, 5, N+1)
	#t = np.arange(0, 5+dt, dt)
	y = forward_euler(dydt, t, y0)
	err[k] = np.abs(y[-1] - ytrue(5))

A1 = y.reshape(-1, 1)
A2 = err.reshape(1, -1)

pfit = np.polyfit(np.log(dt_vals), np.log(err), 1)
A3 = pfit[0]

fig, ax = plt.subplots()
ax.loglog(dt_vals, err, 'k.', markersize=20, label='Forward Euler Error')
ax.loglog(dt_vals, 2.8*dt_vals, 'k--', linewidth=2, label=r'O($\Delta t$) trend line')

# b - Heun's method
def heun(f, t, y0):
	dt = t[2]-t[1]
	y = np.zeros(len(t))
	y[0] = y0
	for k in range(len(y)-1):
		y[k+1] = y[k] + 0.5*dt*( \
			f(t[k], y[k]) + f(t[k+1], y[k]+dt*f(t[k], y[k])) )
	return y

err2 = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	t = np.arange(0, 5+dt, dt)
	yh = heun(dydt, t, y0)
	err2[k] = np.abs(yh[-1] - ytrue(5))

					
A4 = yh.reshape(-1, 1)
A5 = err2.reshape(1, -1)

pfit2 = np.polyfit(np.log(dt_vals), np.log(err2), 1)
A6 = pfit2[0]

ax.loglog(dt_vals, err2, 'bd', markersize=10, markerfacecolor='b', \
			label="Heun's Error")
ax.loglog(dt_vals, 0.75*dt_vals**2, 'b--', linewidth=2, \
		label=r'O($\Delta t^2$) trend line')


# c - Adams predictor-corrector method
def adam(f, t, y0):
	dt = t[2]-t[1]
	y = np.zeros(len(t))
	y[0] = y0
	y[1] = y0 + dt*f(t[0] + dt/2, y0 + 0.5*dt*f(t[0], y0))
	for k in range(1,len(y)-1):
		yp = y[k] + 0.5*dt*(3*f(t[k], y[k]) - f(t[k-1], y[k-1]))
		y[k+1] = y[k] + 0.5*dt*( \
			f(t[k+1], yp) + f(t[k], y[k]) )
	return y

err = np.zeros(len(dt_vals))
for k, dt in enumerate(dt_vals):
	t = np.arange(0, 5+dt, dt)
	y = adam(dydt, t, y0)
	err[k] = np.abs(y[-1] - ytrue(5))

					
ax.loglog(dt_vals, err, 'gs', markersize=10, markerfacecolor='g', \
			label="Adam's Predictor-Corrector Error")
ax.loglog(dt_vals, 8*dt_vals**3, 'g--', linewidth=2, \
		label=r'O($\Delta t^3$) trend line')
ax.legend(loc='lower right')
ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'Global error at t=5: $|y_{true}(5) - y_N|$')
ax.set_title('Global error trends for three methods')

fig.savefig('trend_lines_python.png')
	
## Problem 3
# Parameters
a1, a2, b, c, I = 0.05, 0.25, 0.1, 0.1, 0.1
tvals = np.arange(0, 100+0.5, 0.5)

dv1dt = lambda v1, w1, v2, w2, d12: -v1**3 + (1+a1)*v1**2 - a1*v1 - w1 + I + d12*v2
dw1dt = lambda v1, w1, v2, w2: b*v1 - c*w1
dv2dt = lambda v1, w1, v2, w2, d21: -v2**3 + (1+a2)*v2**2 - a2*v2 - w2 + I + d21*v1
dw2dt = lambda v1, w1, v2, w2: b*v2 - c*w2

dydt = lambda t, y, d12, d21: np.array([dv1dt(y[0], y[1], y[2], y[3], d12),
                          dw1dt(y[0], y[1], y[2], y[3]),
                          dv2dt(y[0], y[1], y[2], y[3], d21),
                          dw2dt(y[0], y[1], y[2], y[3])
                         ])

y0 = np.array([0.1, 0, 0.1, 0])

sol1 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0,
				 method='BDF', args=[0, 0], t_eval=tvals)
sol2 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0,
				 method='BDF', args=[0, 0.2], t_eval=tvals)
sol3 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0,
				 method='BDF', args=[-0.1, 0.2], t_eval=tvals)
sol4 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0,
				 method='BDF', args=[-0.3, 0.2], t_eval=tvals)
sol5 = scipy.integrate.solve_ivp(dydt, [tvals[0], tvals[-1]], y0,
				 method='BDF', args=[-0.5, 0.2], t_eval=tvals)

A14 = (sol1.y[(0, 2, 1, 3), :]).T
A15 = (sol2.y[(0, 2, 1, 3), :]).T
A16 = (sol3.y[(0, 2, 1, 3), :]).T
A17 = (sol4.y[(0, 2, 1, 3), :]).T
A18 = (sol5.y[(0, 2, 1, 3), :]).T



plt.show()

