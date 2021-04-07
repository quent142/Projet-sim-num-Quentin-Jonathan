import matplotlib.pyplot as plt
import numpy as np

def dudt(u,v,A,B) :
    der_u = A + v*u**2 - B*u - u
    return der_u

def dvdt(u,v,A,B) :
    der_v = B*u - v*u**2
    return der_v

h = 1
k = 0.1
A = 1
B = 3
D_u = 10

x = np.linspace(0,50,int(50/h) + 1)
y = np.zeros(int(50/h) + 1)
z = np.zeros(int(50/h) + 1)
plt.xlim(0,1)
plt.ylim(0,5)
line1, = plt.plot(x,y)
line2, = plt.plot(x,z)
for i in range(1000) :
    y_g = np.append(y[1 :],y[-1])
    y_d = np.append(y[0],y[: -1])
    z_g = np.append(z[1 :],z[-1])
    z_d = np.append(z[0],z[: -1])
    y_ = y + k*(dudt(y,z,A,B) + D_u*(y_g - 2*y + y_d)/h**2)
    z_ = z + k*(dvdt(y,z,A,B) + 10*(z_g - 2*z + z_d)/h**2)
    y = y_
    z = z_
    plt.pause(0.1)
    
    line1.set_ydata(y)
    line2.set_ydata(z)
