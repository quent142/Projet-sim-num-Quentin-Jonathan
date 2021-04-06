import numpy as np
import matplotlib.pyplot as plt

def func(A,B,u_0,v_0, t_f, h) :
    u = [u_0]
    v = [v_0]
    t = [0]
    der_u = []
    der_v = []
    while t[-1] < t_f :
        dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
        der_u.append(dudt)
        dvdt = B*u[-1] - v[-1]*u[-1]**2
        der_v.append(dvdt)
        u.append(u[-1] + h*dudt)
        v.append(v[-1] + h*dvdt)
        t.append(t[-1] + h)
    dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
    der_u.append(dudt)
    dvdt = B*u[-1] - v[-1]*u[-1]**2
    der_v.append(dvdt)
    return u,v,t,der_u,der_v

fig1 = plt.figure()
fig1.suptitle('Brusselator 0D')
ax1 = fig1.add_subplot(121)
ax1.plot(t,a,label = 'composant u')
ax1.plot(t,b,label = 'composant v')
plt.legend()

ax2 = fig1.add_subplot(122)
ax2.plot(b,a, label = 'u en fonction de v')
plt.legend()
