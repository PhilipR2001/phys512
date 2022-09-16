import numpy as np
import matplotlib.pyplot as plt

def derivative(f,x,dx):

	if callable(f)==False:
		raise Exception("f must be a callable function")


	D1=(f(x+dx)-f(x-dx))/(2*dx)

	D2=(f(x+2*dx)-f(x-2*dx))/(4*dx)

	return (4*D1-D2)/3

x=1
dx=10**np.linspace(-14,0,201)


#First, with f(x)=exp(x):
plt.loglog(dx,np.abs(derivative(np.exp,x,dx)-np.exp(x)),label="$exp(x)$")

#Next, with f(x)=exp(0.01x):
def func(x):
	return np.exp(0.01*x)

plt.plot(dx,np.abs(derivative(func,x,dx)-0.01*np.exp(0.01*x)),label="$exp(0.01x)$")

plt.legend()
plt.title("Error on derivative for two functions")
plt.ylabel("Error")
plt.xlabel("$dx$")
plt.show()
