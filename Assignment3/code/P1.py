import numpy as np
import matplotlib.pyplot as plt


def rk4_step(fun,x,y,h):

	k1=h*fun(x,y)
	k2=h*fun(x+h/2,y+k1/2)
	k3=h*fun(x+h/2,y+k2/2)
	k4=h*fun(x+h,y+k3)
	
	return y+(k1+2*k2+2*k3+k4)/6


def f(x,y):
	return y/(1+x**2)


n=201

x=np.linspace(-20,20,n)
y=np.zeros_like(x)

y[0]=1

for i in range(n-1):
	
	h=x[i+1]-x[i]
	y[i+1]=rk4_step(f,x[i],y[i],h)
	
plt.plot(x,y-np.exp(np.arctan(20))*np.exp(np.arctan(x)))
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Error using simple RK4 (n=200)")
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.show()

def rk4_stepd(fun,x,y,h):

	y1=rk4_step(fun,x,y,h)
	y2=rk4_step(fun,x+h/2, rk4_step(fun,x,y,h/2) , h/2)

	return (16*y2-y1)/15



n=68

x=np.linspace(-20,20,n)
y=np.zeros_like(x)
y[0]=1

for i in range(n-1):
	
	h=x[i+1]-x[i]	
	y[i+1]=rk4_stepd(f,x[i],y[i],h)


plt.plot(x,(y-np.exp(np.arctan(20))*np.exp(np.arctan(x))))
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Error using modified RK4 (n=67)")
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.show()
