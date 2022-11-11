import numpy as np
import matplotlib.pyplot as plt

l=14.33
N=200
x=np.arange(N)

y=np.sin(2*np.pi*l*x/N)

fy=np.fft.fft(y)
k=np.linspace(0,N,N)
afy=((1-np.exp(-2*1J*np.pi*(k-l)))/(1-np.exp(-2*1J*np.pi*(k-l)/N))-(1-np.exp(-2*1J*np.pi*(k+l)))/(1-np.exp(-2*1J*np.pi*(k+l)/N)))/(2*1J)



plt.plot(fy,label='Numerical Fourier Transform')
plt.plot(afy,label='Analytical Fourier Transform')
plt.xlabel('k')
plt.ylabel('F(k)')
plt.legend()
#plt.show()
plt.clf()
####

window=0.5-0.5*np.cos(2*np.pi*x/N)

fynew=np.zeros_like(fy)

for i in range(len(fynew)):
	fynew[i]=1/2*(fy[i]-0.5*fy[i-1]-0.5*fy[i-N+1])

plt.plot(fy,label='Without Window')
plt.plot(np.fft.fft(window*y),label='With Window')
plt.xlabel('k')
plt.ylabel('F(k)')
plt.legend()
plt.show()

#plt.plot(y*window)
plt.plot(np.fft.fft(window*y)-fynew)
plt.xlabel('k')
plt.ylabel('Difference')
plt.show()
