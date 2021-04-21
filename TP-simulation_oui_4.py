#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:50:59 2021

@author: johnny
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def ifft_and_shift(uk):
    return np.fft.ifft(np.fft.ifftshift(uk))

def fft_and_shift(fct, N):
    Delta = 2*math.pi/N
    x = np.linspace(0, (N-1)*Delta, N)
    un = fct(x)
    
    return np.fft.fftshift(np.fft.fft(un))/N

def deriv(fct_k, N, n_der):
    dom = np.linspace(-N/2,(N/2)-1, N)
    L = 2* np.pi
    duk = ((1j * dom * 2 * np.pi/L)**n_der) * fct_k 
    
    return duk

def Euler_av(uk_0, N, fct, n_der):
    ukt = np.zeros((2, N))
    ukt[:,0] = uk_0
    for i in range(-N/2, N/2):
        ukt[:, i+1] = ukt[:, i] + (-c) * deriv(fct, N, n_der)
    return ukt

def f_cond_init(x):
    mu0 = 2*np.pi /5
    sigma0 = 2*np.pi /10
    return np.exp((-(x-mu0)**2)/(2*sigma0**2))

def fct(x):
    mu0 = 2*np.pi /5
    sigma0 = 2*np.pi /10
    c = 2 * np.pi /10
    oui = np.exp((-(x-mu0-c*t)**2)/(2*sigma0**2))
    pass

#je ne sais pas!
c = 2 * np.pi /10
N=40
Delta = 2*math.pi/N
x = np.linspace(0, (N-1)*Delta, N)
uk0 = f_cond_init(x)

