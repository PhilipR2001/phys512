import numpy as np
import matplotlib.pyplot as plt

def lorentzian(N):
	return np.tan(np.pi/2*(np.random.rand(N)))

N=int(1e6)
x=lorentzian(N)


bins=np.linspace(0,5,501)

a,b=np.histogram(x,bins)
a=a/N/(b[1]-b[0])


b=0.5*(b[1:]+b[:-1])


pdf=2/np.pi/(1+b**2)

#plt.plot(b,a)
#plt.plot(b,pdf)
#plt.show()


accept=np.where(1/(1+x**2)*np.random.rand(N)<np.exp(-x))
x_use=x[accept]

print("Efficiency is: ",len(x_use)/N)
a,b=np.histogram(x_use,bins)
a=a/len(x_use)/(b[1]-b[0])
b=0.5*(b[1:]+b[:-1])

plt.plot(b,a,label='Generated Deviate')

plt.plot(b,np.exp(-b),label='Exponential PDF')
plt.legend()
plt.show()


