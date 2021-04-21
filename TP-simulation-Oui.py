#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:54:54 2021

@author: johnny
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#coucou

N = 64

Delta = 2*math.pi/N
x = np.linspace(0, (N-1)*Delta, N)
un = np.exp(np.sin(3*x))


u = np.fft.fft(un) #jsp ce que c'est mais c'est un complexe

uk = np.fft.fftshift(np.fft.fft(un))/N

dom = np.linspace(-N/2,(N/2)-1, N)

uu = np.fft.ifft(np.fft.ifftshift(uk)) 

ax1 = plt.figure()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("exp(sin(3x))")
plt.plot(x, un)
plt.show()

ax2 = plt.figure()
plt.ylabel("fft")
plt.plot(uu)
plt.show()

ax3 = plt.figure()
plt.ylabel("fft+")
plt.plot(dom, uk.real)
plt.plot(dom, uk.imag)
plt.show()

#après il faut faire fftshift() comme dans les slides pour avoir un "pic" (j'ai pas vrmt compris mdr)

