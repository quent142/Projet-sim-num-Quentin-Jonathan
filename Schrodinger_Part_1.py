import numpy as np
import matplotlib.pyplot as plt

#--------Function-----------

def get_g (dt, K, psi):#progression due àtidt
    return np.exp(-1j * dt * K * np.abs(psi)**2) * psi

def get_gk (g):#g in the spectral space
    return np.fft.fftshift(np.fft.fft(g))

def psi_k_plus(gk, dt, dom_k, L):#psi to t+dt in spectral space
    return np.exp(-(1j/2)*dt*(2*np.pi* dom_k /L)**2) * gk

def psi_plus(psi_k_plus):#conversion physical space
    return np.fft.ifft(np.fft.ifftshift(psi_k_plus))

def psi_perig(dom_x,t) : #initial condition
    return (1 - 4*(1 + 2j*t)/(1 + 4*dom_x**2 + 4*t**2))*np.exp(1j*t)

#--------Computing-----------

N = 2**11 #subdivision of x coordonate
dt = 0.01 #time step
K = -1 
L = 12 #width
t_0 = -10 #initial time
t = 10 #final time

#all vectors for the coordonates
dom_x = np.arange(-int(L/2),int(L/2), L/N)
dom_k = np.linspace(-N/2,(N/2)-1, N)
dom_t = np.linspace(t_0, t, int((t-t_0)/dt)+1)

#creation of the matrice that will collect data, and setup of initial conditions
psi = np.zeros((N, int((t-t_0)/dt)+1), dtype= complex)
psi[:,0] = psi_perig(dom_x, t_0)

for i in range(len(dom_t)-1): #solving
   g = get_g(dt, K, psi[:,i])
   gk = get_gk(g)
   psi_kn = psi_k_plus(gk, dt, dom_k, L)
   psi_xn = psi_plus(psi_kn)
   psi[:,i+1] = psi_xn

#conversion to amplitude
amp_psi = np.abs(psi)

#following steps will produce graph
[xx,tt] = np.meshgrid(dom_x, dom_t)
fig = plt.figure()
plt.contourf(xx,tt,np.transpose(amp_psi),levels = np.linspace(0,np.max(amp_psi)))
plt.colorbar()

#same principle but for the analytical solution
psi_t = np.zeros((N,int((t-t_0)/dt)+1),dtype=complex)

for i,t in zip(range(len(dom_t)),dom_t) :
    psi_t[:,i] = psi_perig(dom_x,t)

amp_psi_t = np.abs(psi_t)

fig_ = plt.figure()
plt.contourf(xx,tt,np.transpose(amp_psi_t),levels = np.linspace(0,np.max(amp_psi_t)))
plt.colorbar()



'''
dif = amp_psi - amp_psi_t

fig_dif = plt.figure()
plt.contourf(xx,tt,np.transpose(dif),levels = np.linspace(np.min(dif),np.max(dif)))
plt.colorbar()
'''

fig_tranche_x = plt.figure()
plt.plot(dom_x,amp_psi[:,-1], label = "calculated_x")
plt.plot(dom_x,amp_psi_t[:,-1], label = "true_x")
plt.legend()

fig_tranche_t = plt.figure()
plt.plot(dom_t,amp_psi[int(N/2),:], label = "calculated_t")
plt.plot(dom_t,amp_psi_t[int(N/2),:], label = "true_t")
plt.legend()

fig_bosse = plt.figure()
plt.plot(dom_x,amp_psi[:,int(0.5*(t-t_0)/dt)], label = "calculated_b")
plt.plot(dom_x,amp_psi_t[:,int(0.5*(t-t_0)/dt)], label = "true_b")
plt.legend()


plt.show()

