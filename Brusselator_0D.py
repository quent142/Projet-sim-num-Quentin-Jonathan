import numpy as np
import matplotlib.pyplot as plt

def func(A,B,u_0,v_0, t_f, N) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = np.linspace(0,t_f,N+1)
    for i in range(N) :
        dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
        der_u.append(dudt)
        dvdt = B*u[-1] - v[-1]*u[-1]**2
        der_v.append(dvdt)
        u.append(u[-1] + h*dudt)
        v.append(v[-1] + h*dvdt)
    return u,v,t

fig1 = plt.figure()
fig1.suptitle('Brusselator 0D')
ax1 = fig1.add_subplot(121)
ax1.plot(t,u,label = 'composant u')
ax1.plot(t,v,label = 'composant v')
plt.legend()

ax2 = fig1.add_subplot(122)
ax2.plot(v,u, label = 'u en fonction de v')
plt.legend()
