import numpy as np
import matplotlib.pyplot as plt

def ndiff(fun,x,full=False):
	
	eps=1e-16

	delta=1e-2

	fthird=(fun(x+2*delta)-fun(x-2*delta)+2*fun(x-delta)-2*fun(x+delta))/(2*delta**3)
	
	if fthird==0:
		fthird=fun(x)

	dx=(np.abs(0.75*fun(x)/fthird*eps))**(1/3)

	fprime=(fun(x+dx)-fun(x-dx))/(2*dx)

	error = np.abs(fthird/6*dx**2 + eps*fun(x)/(2*dx))

	if full:
		return (fprime,dx,error)

	else:
		return fprime


