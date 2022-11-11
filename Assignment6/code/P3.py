import numpy as np
import matplotlib.pyplot as plt


def safeConvolution(f,g):

	fNew=np.zeros(2*len(f))
	fNew[len(f):]=f

	gNew=np.zeros(2*len(g))
	gNew[len(f):]=g

	F=np.fft.fft(fNew)
	G=np.fft.fft(gNew)

	return np.fft.ifft(F*G)[len(f):]

def conv(f,g):
	F=np.fft.fft(f)
	G=np.fft.fft(g)
	return np.fft.ifft(F*G)

x=np.linspace(-3,3,100)
y=np.exp(-0.5*x**2)
xp=np.linspace(-3,3,200)

plt.plot(np.roll(safeConvolution(y,y),0),label='Convolution With Padding')
plt.plot(np.roll(conv(y,y),0),label='Regular Convolution')
plt.xlabel('x')
plt.ylabel('$f*f$')
plt.legend()
plt.show()
