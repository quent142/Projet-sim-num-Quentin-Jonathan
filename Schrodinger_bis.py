import numpy as np
import matplotlib.pyplot as plt

#--------Function-----------

def get_g (dt, K, psi):#progression due à dt
    return np.exp(-1j * dt * K * np.abs(psi)**2) * psi

def get_gk (g, N):#g mais dans l'espace spectral
    return np.fft.fftshift(np.fft.fft(g))#/(N)

def psi_k_plus(gk, N, dt, dom_k, L):#psi en t+dt dans l'espace spectral
    return np.exp(-(1j/2)*dt*(2*np.pi* dom_k /L)**2) * gk

def psi_plus(psi_k_plus):#psi_k mais dans l'espace physique
    return np.fft.ifft(np.fft.ifftshift(psi_k_plus))

def psi_init(dom_x,t_0):
    return (1 - 4*(1 + 2j*t_0)/(1 + 4*dom_x**2 + 4*t_0**2))*np.exp(1j*t_0)

def psi_truth(dom_x,t) :
    return (1 - 4*(1 + 2j*t)/(1 + 4*dom_x**2 + 4*t**2)**2)*np.exp(1j*t)

#--------Computing-----------

N = 1024
dt = 0.01
K = -1
L = 200
t_0 = 0
t = 20

dom_x = np.arange(-int(L/2),int(L/2), L/N)
dom_k = np.linspace(-N/2,(N/2)-1, N)
dom_t = np.linspace(t_0, t, int((t-t_0)/dt)+1)

psi = np.zeros((N, int((t-t_0)/dt)+1), dtype= complex)
psi[:,0] = psi_init(dom_x, t_0)

for i in range(len(dom_t)-1):
   g = get_g(dt, K, psi[:,i])
   
   gk = get_gk(g, N)
   psi_kn = psi_k_plus(gk, N, dt, dom_k, L)
   psi_xn = psi_plus(psi_kn)
   psi[:,i+1] = psi_xn
   
amp_psi = np.abs(psi)

[xx,tt] = np.meshgrid(dom_x, dom_t)
fig = plt.figure()
plt.contourf(xx,tt,np.transpose(amp_psi),levels = np.linspace(0,np.max(amp_psi)))
plt.colorbar()

psi_t = np.zeros((N,int((t-t_0)/dt)+1),dtype=complex)

for i,t in zip(range(len(dom_t)),dom_t) :
    psi_t[:,i] = psi_truth(dom_x,t)

amp_psi_t = np.abs(psi_t)

fig_ = plt.figure()
plt.contourf(xx,tt,np.transpose(amp_psi_t),levels = np.linspace(0,np.max(amp_psi_t)))
plt.colorbar()
plt.show()

dif = amp_psi - amp_psi_t

fig_dif = plt.figure()
plt.contourf(xx,tt,np.transpose(dif),levels = np.linspace(np.min(dif),np.max(dif)))
plt.colorbar()
plt.show()