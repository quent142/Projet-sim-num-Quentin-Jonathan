import matplotlib.pyplot as plt
import numpy as np
import random as rd

def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

h = 1
k = 0.1
A = 1
B = 3
D_u = 10
N = int(50/h) + 1

x = np.linspace(0,50,N)
u = np.zeros(N)
v = np.zeros(N)
'''
for i in range(N) :
    u[i] = rd.uniform(0,5)
    v[i] = rd.uniform(0,5)
'''
time = 10
plt.xlim(0,1)
plt.ylim(0,5)
line1, = plt.plot(x,u, label = 'composant u')
line2, = plt.plot(x,v, label = 'composant v')
plt.legend()
for i in range(int(time/k)) :
    u_g = np.append(u[1 :],u[-1])
    u_d = np.append(u[0],u[: -1])
    v_g = np.append(v[1 :],v[-1])
    v_d = np.append(v[0],v[: -1])
    u_ = u + k*(dudt(u,v,A,B) + D_u*(u_g - 2*u + u_d)/h**2)
    v_ = v + k*(dvdt(u,v,A,B) + 10*(v_g - 2*v + v_d)/h**2)
    u = u_
    v = v_
    plt.pause(k)
    
    line1.set_ydata(u)
    line2.set_ydata(v)
