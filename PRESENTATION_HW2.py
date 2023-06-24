#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.integrate
from matplotlib.pyplot import figure


# In[2]:


import sys
np.set_printoptions(threshold=sys.maxsize)


# ### Problem 1 

# In[3]:


# Define the function
def rhsfunc(x, y, ep):
    f1 = y[1]
    f2 = (x**2 - ep) * y[0]
    
    return np.array([f1,f2])


# In[4]:


# Define some constants 
K = 1
L = 4
ep_start = 0
tol = 10**(-6) 

xp = [-L, L]
x_evals = np.linspace(-L,L,20*L+1) 


# In[5]:


ep = ep_start
dep = K/100 


# In[6]:


# Define our initial conditions 
y0 = np.array([1,((K*L**2)-ep)**.5])


# In[7]:


A6 = np.array([])
A = {}

for modes in range(5):
    ep = ep_start
    dep = K/100
    
    
    for j in range(1000): # using for loop to make sure it stops. 
        sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc(x,y,ep), xp, y0, t_eval = x_evals)
        y_sol = sol.y[0,:] # f1 
        y_sol_dx = sol.y[1,:] #f2 
        
        # Normalize eigenfunction
        y_sol_norm = y_sol/(np.sqrt(scipy.integrate.trapz(y_sol**2, x_evals)))

        if np.abs(y_sol_dx[-1] + (((K*L**2)-ep)**.5)*y_sol[-1]) < tol:
#             print('We got the eigenvalue:', ep)
            A6 = np.append(A6, ep)
            break 

        if (-1)**(modes)*(y_sol_dx[-1] + (((K*L**2)-ep)**.5)*y_sol[-1]) > tol:
            ep = ep + dep # Increase 
        else: 
            ep = ep - dep/2 # Decrease 
            dep = dep/2 # Cut dep in half to make sure we converge 

        y0 = np.array([1,((K*L**2)-ep)**.5])
        
    A[modes] = y_sol_norm
    
    ep_start = ep + 0.1
    

#     plt.plot(sol.t, y_sol_norm, linewidth=2)
#     plt.plot(sol.t, 0*sol.t, 'k')


# In[8]:


A1 = np.abs(A[0].copy()).reshape(81,1)
A2 = np.abs(A[1].copy()).reshape(81,1)
A3 = np.abs(A[2].copy()).reshape(81,1)
A4 = np.abs(A[3].copy()).reshape(81,1)
A5 = np.abs(A[4].copy()).reshape(81,1)


# In[9]:


A6 = A6.reshape(1,5)


# In[10]:


L = 4
x = np.linspace(-L, L, 20*L+1)
t = np.linspace(0,5,100)


# In[11]:


sol = A[2].reshape(81,1) * ((np.cos(A6[:,1]) * t)/2).reshape(1,100)


# In[13]:


# In order to do colors, need to import another package
from mpl_toolkits import mplot3d
from matplotlib import cm


# In[14]:


fig4 = plt.figure(figsize=(10,10))
ax4 = plt.axes(projection = '3d')

X, T = np.meshgrid(x, t)
surf = ax4.plot_surface(X, T, sol.T.real, cmap = cm.hsv, rstride=1, cstride=1)
fig4.colorbar(surf, orientation = 'horizontal')
ax4.view_init(-140, 30)


# ax4.set_title('Time evolution of the second mode', fontsize = 16)
# ax4.set_zlabel(r'$\psi_2$', fontsize=14, rotation = 90)
# ax4.set_xlabel(r'$x-value$', fontsize=14, rotation=60)
# ax4.set_ylabel(r'$time$', fontsize=14, rotation=60)

# fig4.savefig("-14030.pdf")


# In[15]:


fig4 = plt.figure(figsize=(10,10))

ax4 = plt.axes(projection = '3d')

X, T = np.meshgrid(x, t)
surf = ax4.plot_surface(X, T, sol.T.real, cmap = cm.hsv, rstride=1, cstride=1)
fig4.colorbar(surf, orientation = 'horizontal')
ax4.view_init(45, 45)
# ax4.view_init(-140, 30)


ax4.set_title('Time evolution of the second mode', fontsize = 16)
ax4.set_zlabel(r'$\psi_2$', fontsize=14, rotation = 90)
ax4.set_xlabel(r'$x-value$', fontsize=14, rotation=60)
ax4.set_ylabel(r'$time$', fontsize=14, rotation=60)

fig4.savefig("-14030.pdf")


# In[16]:


fig5, ax5 = plt.subplots()

X, T = np.meshgrid(x, t)
surf = ax5.contourf(X, T, sol.T.real)
fig5.colorbar(surf)

ax5.set_title('Contour plot of the time evolution of the second mode', fontsize = 16)
ax5.set_xlabel(r'$x-value$', fontsize=14)
ax5.set_ylabel(r'$time$', fontsize=14)

fig5.savefig("contour.pdf")

