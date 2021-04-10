import numpy as np
import matplotlib.pyplot as plt

'Paramètres'
A = 1
B = 1.3
Du= 0.1
Dv= 10
N = 5000
L = 50
t = np.arange(N)
x = np.arange(L)
u = np.zeros((L,N))
v = np.zeros((L,N))


for p in range (20):
    k = (0.005)*(p+5)
    h = (0.075)*(p+5)
    lu= k*Du/(h**2)
    lv= k*Dv/(h**2)
    'conditions initiales'
    u[:,0] = np.random.random_sample(L)
    v[:,0] = np.random.random_sample(L)

    'Définition de la matrice d iteration '
    Hu = np.eye(L,L,k=-1)*(-lu) + np.eye(L,L)*(2*lu+1) + np.eye(L,L,k=1)*(-lu)
    Hv = np.eye(L,L,k=-1)*(-lv) + np.eye(L,L)*(2*lv+1) + np.eye(L,L,k=1)*(-lv)

    'Conditions aux bords'
    Hu[0,0] = 3
    Hu[0,1] =-4
    Hu[0,2] = 1

    Hu[L-1,L-1] = 3
    Hu[L-1,L-2] =-4
    Hu[L-1,L-3] = 1

    Hv[0,0] = 3
    Hv[0,1] =-4
    Hv[0,2] = 1

    Hv[L-1,L-1] = 3
    Hv[L-1,L-2] =-4
    Hv[L-1,L-3] = 1

    'iteration du système'
    for j in range (N-1):
        fu = k*( v[:,j] * u[:,j]**2 - B*u[:,j] - u[:,j] + A) + u[:,j]
        fv = k*(-v[:,j] * u[:,j]**2 + B*u[:,j]) + v[:,j]
        'conditions aux bords'
        fu[0]  =0
        fu[L-1]=0
        fv[0]  =0
        fv[L-1]=0
        
        u[:,j+1] = np.linalg.solve(Hu,fu)
        v[:,j+1] = np.linalg.solve(Hv,fv)
 
    'Il faudrait ploter sur une ligne de longueur L la concentration en u et en v'
    plt.figure(p)
    plt.plot(x,u[:,N-1])
    plt.plot(x,v[:,N-1])
    
    
    
'''


for j in range (10):
    plt.figure(j)
    plt.plot(x,u[:,40*j])
    plt.plot(x,v[:,40*j])



for j in range (10):
    plt.figure(j)
    plt.plot(t,u[4*j,:])
    plt.plot(t,v[4*j,:])


    
for j in range (10):
    plt.figure(j)    
    plt.plot(u[:,40*j],v[:,40*j])       
 
    
    
for j in range (10):
    plt.figure(j)    
    plt.plot(u[4*j,:],v[4*j,:])

    
'''