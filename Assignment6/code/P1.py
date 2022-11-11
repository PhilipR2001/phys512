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



x=np.arange(200)
xShift=shift(x,50)

plt.plot(x,label='Original')
plt.plot(xShift,label='Shifted')
plt.legend()
plt.show()
