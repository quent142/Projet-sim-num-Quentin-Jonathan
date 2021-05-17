import numpy as np
import matplotlib.pyplot as plt

#définition des dérivées du système
def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

'''
def eul_av(A,B,u_0,v_0, t_f, N) :
    h = t_f/N
    u = [u_0]
    v = [v_0]
    t = np.linspace(0,t_f,N+1)
    for i in range(N) :
        u.append(u[-1] + h*dudt(u[-1],v[-1],A,B))
        v.append(v[-1] + h*dvdt(u[-1],v[-1],A,B))
    return u,v,t
'''

def rk4(A,B,u_0,v_0, t_f, N) : #introduction de la méthode de Runge-Kutta
    h = t_f/N #pas de temps
    
    #conditions initiales
    u = [u_0]
    v = [v_0]
    
    t = np.linspace(0,t_f,N+1)
    for i in range(N) : #itération
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

#introduction de nos paramètres
A = 1
n_b = 25 #nombre de subdivision que l'on va appliquer à B
B = np.linspace(0, 5, n_b+1) #création de tous les B à analyser
for i in range(n_b):
    B[i] = int(B[i]*1000)/1000 #arrondi au millième
t_f = 100 #temps final

#conditions initiales
u_0 = 0
v_0 = 0

N = 10000 #nombre d'itérations

#ce seront nos matrices qui acceuilleront toutes les coordonnées de notre simulation
U = []
V = []

for bi in B: #génération des données
    a, b, t = rk4(A, bi, u_0, v_0, t_f, N)
    U.append(a)
    V.append(b)

 #--------

#création du graphe animé
fig1 = plt.figure()
fig1.suptitle('Brusselator 0D')
ax1 = fig1.add_subplot(121)  
ax1.set_xlim(0,t_f)  
hl, = ax1.plot([],[])
h2, = ax1.plot([], [])

ax2 = fig1.add_subplot(122)
h3, = ax2.plot([],[])

def update_line(h, new_datax, new_datay): #fonction pour mettre à jour le graphe
    h.set_xdata(new_datax)
    h.set_ydata(new_datay)
    

for i in range(n_b+1): #lancement de l'animation
    hl.axes.set_ylim(0, max(max(U[i])+0.5, max(V[i]))+0.5)
    h3.axes.set_xlim(0, max(U[i])+0.5)
    h3.axes.set_ylim(0, max(V[i])+0.5)
    
    hl.axes.set_xlabel("B="+str(B[i]))
    update_line(hl, t, U[i])
    update_line(h2, t, V[i])
    
    update_line(h3, U[i],V[i])
    
    plt.draw() #mise à jour du plot
    plt.pause(0.2) #intervale entre 2 images
