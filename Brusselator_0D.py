import numpy as np
import matplotlib.pyplot as plt

def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

def eul_av(A,B,u_0,v_0, t_f, N) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = np.linspace(0,t_f,N+1)
    for i in range(N) :
        dudt = A + v[-1]*u[-1]**2 - B*u[-1] - u[-1]
        dvdt = B*u[-1] - v[-1]*u[-1]**2
        u.append(u[-1] + h*dudt)
        v.append(v[-1] + h*dvdt)
    return u,v,t

def rk4(A,B,u_0,v_0, t_f, h) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = np.linspace(0,t_f,N+1)
    for i in range(N) :
        k1 = h*dudt(u[-1],v[-1],A,B)
        j1 = h*dvdt(u[-1],v[-1],A,B)
        k2 = h*dudt(u[-1] + k1/2,v[-1] + j1/2,A,B)
        j2 = h*dvdt(u[-1] + k1/2,v[-1] + j1/2,A,B)
        k3 = h*dudt(u[-1] + k2/2,v[-1] + j2/2,A,B)
        j3 = h*dvdt(u[-1] + k2/2,v[-1] + j2/2,A,B)
        k4 = h*dudt(u[-1] + k3,v[-1] + j3,A,B)
        j4 = h*dvdt(u[-1] + k3,v[-1] + j3,A,B)
        u.append(u[-1] + (k1 + 2*k2 + 2*k3 + k4)/6)
        v.append(v[-1] + (j1 + 2*j2 + 2*j3 + j4)/6)
    return u,v,t



#------
A = 1
n_b = 26
B = np.linspace(0, 5, n_b)
print(B)
for i in range(n_b):
    B[i] = int(B[i]*1000)/1000
t_f = 100
u_0 = 0
v_0 = 0
N = 10000
NULL = np.zeros(N+1)

U = []
V = []

for bi in B:
    a, b, t = (A, bi, u_0, v_0, t_f, N)
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
    plt.pause(0.1)
