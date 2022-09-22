import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec


sigma=1
eps=1
R=1

### USING SCIPY ###

x=np.linspace(0,4,1001)

def integrand(theta):

	return sigma/(2*eps)*(np.sin(theta)*(x-np.cos(theta)))/(1-2*x*np.cos(theta)+x**2)**(3/2)


E_scipy=quad_vec(integrand,0,np.pi)[0]

E_real=np.zeros_like(x)
E_real[251:]=sigma/(eps*x[251:]**2)

plt.plot(x,E_scipy)
plt.xlabel("$z/R$")
plt.ylabel("Error")
plt.title("E-field using ScipPy")
#plt.show()
plt.clf()

plt.plot(np.delete(x,250),np.delete(E_scipy-E_real,250))
plt.title("Error on the integral using SciPy.quad")
plt.xlabel("$z/R$")
plt.ylabel("Error")

#plt.show()
plt.clf()


### USING SIMPSON ###


theta=np.linspace(0,np.pi,1001)

def integrand2(theta,x):

	return sigma/(2*eps)*(np.sin(theta)*(x-np.cos(theta)))/(1-2*x*np.cos(theta)+x**2)**(3/2)


def simpson(x):
	y=np.zeros_like(x)
	for k in range(len(x)):
		tmp=integrand2(theta,x[k])

		y[k]=(theta[1]-theta[0])/(3)*(tmp[0]+tmp[-1]+4*np.sum(tmp[1:-1:2])+2*np.sum(tmp[2:-1:2]))

	return y

x=np.delete(np.linspace(0,4,1001),250)
y=simpson(x)


plt.plot(x,y-np.delete(E_real,250))
plt.title("Error on integral using Simpson's method")
plt.xlabel("$z/R$")
plt.ylabel("Error")
#plt.show()

