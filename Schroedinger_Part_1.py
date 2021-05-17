import numpy as np
import matplotlib.pyplot as plt

#--------Function-----------

def get_g (dt, K, psi):#progression due à dt
    return np.exp(-1j * dt * K * np.abs(psi)**2) * psi

def get_gk (g):#g mais dans l'espace spectral
    return np.fft.fftshift(np.fft.fft(g))

def psi_k_plus(gk, dt, dom_k, L):#psi en t+dt dans l'espace spectral
    return np.exp(-(1j/2)*dt*(2*np.pi* dom_k /L)**2) * gk

def psi_plus(psi_k_plus):#psi_k mais dans l'espace physique
    return np.fft.ifft(np.fft.ifftshift(psi_k_plus))

def psi_init(dom_x):
    return 0.5 + 0.01 * np.cos(2*np.pi * dom_x / 40)

#--------Computing-----------

N = 2**10
dt = 0.01
K_neg = -1
K_pos = 1
L = 40
t = 200

dom_x = np.arange(-int(L/2),int(L/2), L/N)
dom_k = np.linspace(-N/2,(N/2)-1, N)
dom_t = np.linspace(0, t, int(t/dt)+1)

psi_neg = np.zeros((N, int(t/dt)+1), dtype= complex)
psi_neg[:,0] = psi_init(dom_x)

psi_pos = np.zeros((N, int(t/dt)+1), dtype= complex)
psi_pos[:,0] = psi_init(dom_x)

for i in range(len(dom_t)-1):
   g = get_g(dt, K_neg, psi_neg[:,i])
   gk = get_gk(g)
   psi_kn = psi_k_plus(gk, dt, dom_k, L)
   psi_xn = psi_plus(psi_kn)
   psi_neg[:,i+1] = psi_xn
   
   g = get_g(dt, K_pos, psi_pos[:,i])
   gk = get_gk(g)
   psi_kn = psi_k_plus(gk, dt, dom_k, L)
   psi_xn = psi_plus(psi_kn)
   psi_pos[:,i+1] = psi_xn
   
amp_psi_pos = np.abs(psi_pos)
amp_psi_neg = np.abs(psi_neg)

[xx,tt] = np.meshgrid(dom_x, dom_t)


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.set_title("K=-1")   
cf1 = ax1.contourf(xx,tt,np.transpose(amp_psi_neg))

ax2 = fig1.add_subplot(122)
ax2.set_title("K=1")
cf2 = ax2.contourf(xx,tt,np.transpose(amp_psi_pos))

fig1.colorbar(cf1, ax=ax1)
fig1.colorbar(cf2, ax=ax2)