import matplotlib.pyplot as plt
import numpy as np
import random as rd

#définition des fonctions F, vu que c'est les mêmes que sur brusselator 0D, on garde la même notation
def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

#définition des paramètres
h = 1 #pas d'espace
k = 0.01 #pas de temps
A = 1
B = 3
D_u = 10
D_v = 10
N = int(50/h) + 1
time = 15 #temps final
T = int(time/k)

#discrétisation
x = np.linspace(0,50,N)

#création des matrices regroupant nos données
u = np.zeros((N,T+1))
v = np.zeros((N,T+1))

#conditions initiales aléatoires
for i in range(N) :
    u[i,0] = rd.uniform(0,5)
    v[i,0] = rd.uniform(0,5)
    
#respect des conditions aux bords
u[0,0] = u[1,0]
u[-1,0] = u[-2,0]
v[0,0] = v[1,0]
v[-1,0] = v[-2,0]

#définition des matrices et constantes utilisées
cst_u = D_u*k/h**2
cst_v = D_v*k/h**2
A_u = (1+2*cst_u)*np.eye(N) - cst_u*np.eye(N,k=-1) - cst_u*np.eye(N,k=1)
A_v = (1+2*cst_v)*np.eye(N,N) - cst_v*np.eye(N,k=-1) - cst_v*np.eye(N,k=1)

#modification pour respecter les conditions aux bords
A_u[0,0],A_u[0,1],A_u[-1,-1],A_u[-1,-2] = 1,-1,1,-1
A_v[0,0],A_v[0,1],A_v[-1,-1],A_v[-1,-2] = 1,-1,1,-1

for i in range(T) :
    #création du vecteur pour la résolution du système linéaire à venir
    u_ = u[:,i]
    v_ = v[:,i]
    B_u = u_ + k*(A + v_*u_**2 - B*u_ - u_)
    B_v = v_ + k*(B*u_ - v_*u_**2)
    
    #modification pour respecter les conditions aux bords
    B_u[0],B_u[-1],B_v[0],B_v[-1] = 0,0,0,0 
    
    #résolution du système linéaire
    u[:,i+1] = np.linalg.solve(A_u,B_u)
    v[:,i+1] = np.linalg.solve(A_v,B_v)
 
#définition du graphe
plt.xlabel("t = 0")
plt.xlim(0,50)
if u.max() > v.max() :
    plt.ylim(-1,u.max()+1)
else :
    plt.ylim(-1,v.max()+1)
line1, = plt.plot(x,u[:,0], label = 'u')
line2, = plt.plot(x,v[:,0], label = 'v')
plt.legend(bbox_to_anchor=(1.0001, 1),loc='upper left')

#animation du graphe
for i in range(T+1) :
    plt.pause(k)
    plt.xlabel("t = {}".format(int(100*i*k)/100))
    line1.set_ydata(u[:,i])
    line2.set_ydata(v[:,i])
    plt.draw()
