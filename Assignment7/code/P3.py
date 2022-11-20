import numpy as np
import matplotlib.pyplot as plt



N=int(1e6)

u=np.random.rand(N)
v=2*np.exp(-1)*np.random.rand(N)
r=v/u

accept=np.where(u<np.exp(-0.5*r))

r_use=r[accept]

bins=np.linspace(0,5,501)
a,b=np.histogram(r_use,bins)
a=a/len(r_use)/(b[1]-b[0])
b=0.5*(b[1:]+b[:-1])

plt.plot(b,a,label='Generated Deviate')
plt.plot(b,np.exp(-b),label='Exponential PDF')
plt.legend()
plt.show()

print(len(r_use)/N)



