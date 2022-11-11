import numpy as np
import matplotlib.pyplot as plt


def shift(array,n):
	
	n=n%len(array)	
	delta_n=np.zeros_like(array)
	delta_n[n]=1
	
	new=np.zeros_like(array)

	for i in range(len(array)):
		for j in range(len(array)):
			new[i]=new[i]+delta_n[j]*array[i-j]

	return new 



def correlation(f,g):

	F=np.fft.fft(f)
	G=np.fft.fft(g)

	return np.fft.ifft(F*np.conjugate(G))


x=np.linspace(-3,3,200)
y=np.exp(-0.5*x**2)

plt.plot(np.roll(correlation(y,y),100),label='Unshifted (n=0)')
plt.plot(np.roll(correlation(y,shift(y,50)),100),label='Shifted (n=50)')
plt.xlabel('x')
plt.ylabel('Correlation')

plt.legend()
plt.show()
