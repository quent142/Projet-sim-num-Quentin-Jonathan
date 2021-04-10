import matplotlib.pyplot as plt
import numpy as np
import random as rd

#définition des dérivée partielles par rapport au temps
def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

#définition des paramètres
h = 1
k = 0.1
A = 1
B = 3
D_u = 10
D_v = 10
N = int(50/h) + 1
time = 30

#discrétisation
x = np.linspace(0,50,N)
u = np.zeros(N)
v = np.zeros(N)
#conditions initiales aléatoires
for i in range(N) :
    u[i] = rd.uniform(0,5)
    v[i] = rd.uniform(0,5)
u[0] = u[-1]
u[-1] = u[-2]
v[0] = v[1]
v[-1] = v[-2]

#définition du graphe
plt.xlim(0,50)
plt.ylim(-1,5)
line1, = plt.plot(x,u, label = 'composant u')
line2, = plt.plot(x,v, label = 'composant v')
plt.legend()

#définition des matrices et constantes utilisées
cst_u = D_u*k/h**2
cst_v = D_v*k/h**2
A_u = (1+2*cst_u)*np.eye(N) - cst_u*np.eye(N,k=-1) - cst_u*np.eye(N,k=1)
A_v = (1+2*cst_v)*np.eye(N,N) - cst_v*np.eye(N,k=-1) - cst_v*np.eye(N,k=1)

#modification pour respecter les conditions initiales
A_u[0,0],A_u[0,1],A_u[-1,-1],A_u[-1,-2] = 1,-1,1,-1
A_v[0,0],A_v[0,1],A_v[-1,-1],A_v[-1,-2] = 1,-1,1,-1

for i in range(int(time/k)) :
    #création du vecteur constant pour la résolution du système linéaire à venir
    B_u = u + k*(A + v*u**2 - B*u - u)
    B_v = v + k*(B*u - v*u**2)
    
    #modification pour respecter les conditions initiales
    B_u[0],B_u[-1],B_v[0],B_v[-1] = 0,0,0,0 
    
    #résolution du système linéaire
    u = np.linalg.solve(A_u,B_u)
    v = np.linalg.solve(A_v,B_v)
    
    #modification du plot
    plt.pause(k)
    line1.set_ydata(u)
    line2.set_ydata(v)
