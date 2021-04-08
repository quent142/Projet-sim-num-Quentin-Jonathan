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
k = 0.01
A = 1
B = 3
D_u = 10
N = int(50/h) + 1
time = 30

x = np.linspace(0,50,N)
u = np.zeros(N)
v = np.zeros(N)

for i in range(N) :
    u[i] = rd.uniform(0,5)
    v[i] = rd.uniform(0,5)
u[-1] = u [-2]
u[0] = u[1]
v[-1] = v[-2]
v[0] = v[1] 
plt.xlim(0,50)
plt.ylim(-1,5)
line1, = plt.plot(x,u, label = 'composant u')
line2, = plt.plot(x,v, label = 'composant v')
plt.legend()
for i in range(int(time/k)) :
    u_g = np.append(u[1 :],u[-1])
    u_d = np.append(u[0],u[: -1])
    v_g = np.append(v[1 :],v[-1])
    v_d = np.append(v[0],v[: -1])
    K1 = k*(dudt(u,v,A,B) + D_u*(u_g - 2*u + u_d)/h**2)
    J1 = k*(dvdt(u,v,A,B) + 10*(v_g - 2*v + v_d)/h**2)
    K2 = k*(dudt(u + K1/2,v + J1/2,A,B) + D_u*(u_g - 2*u + u_d)/h**2)
    J2 = k*(dvdt(u + K1/2,v + J1/2,A,B) + 10*(v_g - 2*v + v_d)/h**2)
    K3 = k*(dudt(u + K2/2,v + J2/2,A,B) + D_u*(u_g - 2*u + u_d)/h**2)
    J3 = k*(dvdt(u + K2/2,v + J2/2,A,B) + 10*(v_g - 2*v + v_d)/h**2)
    K4 = k*(dudt(u + K3,v + J3,A,B) + D_u*(u_g - 2*u + u_d)/h**2)
    J4 = k*(dvdt(u + K3,v + J3,A,B) + 10*(v_g - 2*v + v_d)/h**2)
    u += (K1 + 2*K2 + 2*K3 + K4)/6
    v += (J1 + 2*J2 + 2*J3 + J4)/6 
    u[-1] = u [-2]
    u[0] = u[1]
    v[-1] = v[-2]
    v[0] = v[1] 
    plt.pause(k)
    line1.set_ydata(u)
    line2.set_ydata(v)
