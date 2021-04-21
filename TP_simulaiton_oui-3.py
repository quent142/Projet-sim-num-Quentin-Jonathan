#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:32:08 2021

@author: johnny
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def ddfoui(fct, N, kf):
    Delta = 2 * math.pi/N
    x = np.linspace(0, (N-1)*Delta, N)
    un = fct(x)
    uk = np.fft.fftshift(np.fft.fft(un))/N
    for i, uki in enumerate(uk):
        if abs(dom[i]) > kf:
            uk[i] = 0
     
    return np.fft.ifft(np.fft.ifftshift(uk))

def fct1(x):
    return np.sin(x) + np.sin(20*x)

def fct2(x):
    r = np.random.random((N,))*2 - [1]*N
    gamma = 1
    return np.sin(x) + gamma * r
    
N = 128
kf = 4

dom = np.linspace(-N/2,(N/2)-1, N)
Delta = 2 * math.pi/N
x = np.linspace(0, (N-1)*Delta, N)
'''
fct = fct1(x)

filtered_fct = ddfoui(fct1, N, kf)

ax3 = plt.figure()
plt.ylabel("base")
plt.plot(dom, fct)
plt.show()

ax3 = plt.figure()
plt.ylabel("filtré")
plt.plot(dom, filtered_fct)
plt.show()
'''

fct_2 = fct2(x)
filtered_fct2 = ddfoui(fct2, N, 2)
filtered_fct2_01 = ddfoui(fct2, N, 10)


ax3 = plt.figure()
plt.ylabel("base")
plt.plot(dom, fct_2)
plt.show()


ax3 = plt.figure()
plt.ylabel("filtré_kf=2")
plt.plot(dom, filtered_fct2)
plt.show()

ax3 = plt.figure()
plt.ylabel("filtré_kf=10")
plt.plot(dom, filtered_fct2_01)
plt.show()
