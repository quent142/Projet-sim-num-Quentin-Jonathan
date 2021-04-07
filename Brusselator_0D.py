import numpy as np
import matplotlib.pyplot as plt

def func(A,B,u_0,v_0, t_f, N) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = np.linspace(0,t_f,N+1)
    der_u = []
    der_v = []
    for i in range(N) :
        dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
        der_u.append(dudt)
        dvdt = B*u[-1] - v[-1]*u[-1]**2
        der_v.append(dvdt)
        u.append(u[-1] + h*dudt)
        v.append(v[-1] + h*dvdt)
    return u,v,t
#------
A = 1
n_b = 26
B = np.linspace(0, 5, n_b)
for i in range(n_b):
    B[i] = int(B[i]*1000)/1000
t_f = 10
u_0 = 0
v_0 = 0
N = 64
NULL = np.zeros(N+1)

U = []
V = []

for bi in B:
    a, b, t = func(A, bi, u_0, v_0, t_f, N)
    U.append(a)
    V.append(b)

 #--------

fig1 = plt.figure()
fig1.suptitle('Brusselator 0D')
ax1 = fig1.add_subplot(121)    
hl, = ax1.plot([],[])
ax1.set_xlim(t_f)
hl2, = ax1.plot(t, NULL)

ax2 = fig1.add_subplot(122)
hl3, = ax2.plot([],[])

def update_line(hl, new_datax, new_datay):
    hl.set_xdata(new_datax)
    hl.set_ydata(new_datay)
    

for i in range(n_b):
    hl.axes.set_ylim(0, max(max(U[i]), max(V[i]))+0.5)
    hl3.axes.set_xlim(0, max(U[i])+0.5)
    hl3.axes.set_ylim(0, max(V[i])+0.5)
    
    hl.axes.set_xlabel("B="+str(B[i]))
    update_line(hl, t, U[i])
    
    
    update_line(hl2, t, V[i])
    
    update_line(hl3, U[i],V[i])
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
