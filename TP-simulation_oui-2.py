#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:21:12 2021

@author: johnny
"""
import numpy as np
import matplotlib.pyplot as plt
import math

N = 128

dom = np.linspace(-N/2,(N/2)-1, N)

L = 2* np.pi
Delta = 2*math.pi/N
x = np.linspace(0, (N-1)*Delta, N)
uz = np.exp(np.sin(3*x)) * np.cos(3*x)*3


def exp_sin3x(x):
    return np.exp(np.sin(3*x))

def fct2(x):
    return abs(np.sin(3*x))
    
def deriv(fct, N, n_der):
    Delta = 2*math.pi/N
    x = np.linspace(0, (N-1)*Delta, N)
    un = fct(x)
    
    uk = np.fft.fftshift(np.fft.fft(un))/N
    duk = 1j * dom * 2 * np.pi * uk /L
    
    return np.fft.ifft(np.fft.ifftshift(duk))

der = deriv(exp_sin3x, N, 1)

dif = np.sqrt(der - uz)



der2 = deriv(fct2, N, 1)
uz2 = (3*np.sin(3*x)*np.cos(3*x))/(abs(np.sin(3*x)))



ax3 = plt.figure()
plt.ylabel("fft+")
plt.plot(dom, der2)
plt.show()

ax1 = plt.figure()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("exp(sin(3x))")
plt.plot(dom, uz2)
plt.show()
'''ax = plt.figure()
plt.title("difference")
plt.plot(x, dif)
plt.show'''

'''
ax1 = plt.figure()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("exp(sin(3x))")
plt.plot(x, uz)
plt.show()

ax3 = plt.figure()
plt.ylabel("fft+")
plt.plot(dom, der)
plt.show()
'''