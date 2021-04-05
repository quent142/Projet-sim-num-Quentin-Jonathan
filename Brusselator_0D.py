import numpy as np
import matplotlib.pyplot as plt

def func(A,B,u_0,v_0, t_f, N) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = [0]
    for i in range(N) :
        dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
        dvdt = B*u[-1] - v[-1]*u[-1]**2
        u.append(u[-1] + h*dudt)
        v.append(v[-1] + h*dvdt)
        t.append(t[-1] + h)
    return u,v,t

a,b,t = func(1,2,0,0,100,1000)
plt.plot(t,a)