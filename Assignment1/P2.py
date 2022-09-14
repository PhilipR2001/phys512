import numpy as np

def ndiff(fun,x,full=False):
	
	eps=1e-16

	dx=(0.75*eps)**(1/3)

	fprime=(fun(x+dx)-fun(x-dx))/(2*dx)

	error = np.abs(fun(x)*(1/6*dx**2 + eps/(2*dx)))

	if full:
		return (fprime,dx,error)

	else:
		return fprime

