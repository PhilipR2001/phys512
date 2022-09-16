import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp

def lorentz(x):
	return 1/(1+x**2)


### POLYNOMIAL ###

def polyfit(x,y,xx):
	yp=0.0*xx

	for i in range(len(x)):
		x_use=np.append(x[:i],x[i+1:])
		x0=x[i]
		mynorm=np.prod(x0-x_use)
		p0=1.0
		for xi in x_use:
			p0=p0*(xx-xi)
		p0=p0/mynorm

		yp=yp+y[i]*p0

	return yp


### SPLINE ###

def splinefit(x,y,xx):
	spline=sp.splrep(x,y)
	return sp.splev(xx,spline)



### RATIONAL ###

def ratfit(x,y,n,m,xx,printCoeffs=False,pinv=False):
	pcols=[x**k for k in range(n+1)]
	pmat=np.vstack(pcols)

	qcols=[-x**k*y for k in range(1,m+1)]
	qmat=np.vstack(qcols)
	mat=np.hstack([pmat.T,qmat.T])
	coeffs=np.linalg.inv(mat)@y
	
	if pinv:
		coeffs=np.linalg.pinv(mat)@y

	p=0
	for i in range(n+1):
		p=p+coeffs[i]*xx**i
	qq=1
	for i in range(m):
		qq=qq+coeffs[n+1+i]*xx**(i+1)
	
	if (printCoeffs):
		print(coeffs[:n+1],coeffs[n+1:])

	return p/qq




n=4
m=5

x=np.linspace(-np.pi/2,np.pi/2,n+m+1)
xx=np.linspace(-np.pi/2,np.pi/2,1001)

y=np.cos(x)

plt.plot(xx,ratfit(x,y,n,m,xx)-np.cos(xx))
plt.plot(xx,polyfit(x,y,xx)-np.cos(xx))
plt.xlabel('x')
plt.ylabel('Error')
plt.legend(['Rational','Polynomial'])
plt.title('Error on Polynomial and Rational Interpolations')
plt.show()

plt.clf()
plt.plot(xx,splinefit(x,y,xx)-np.cos(xx))
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error on Cubic Spline Interpolation')
plt.show()




x=np.linspace(-1,1,n+m+1)
xx=np.linspace(-1,1,1001)

y=lorentz(x)

plt.plot(xx,splinefit(x,y,xx)-lorentz(xx))
plt.plot(xx,polyfit(x,y,xx)-lorentz(xx))
plt.xlabel('x')
plt.ylabel('Error')
plt.legend(['Spline','Polynomial'])
plt.title('Error on Spline and Polynomial Interpolations')
plt.show()

plt.clf()
print("Coefficients using inv:")
plt.plot(xx,ratfit(x,y,n,m,xx,printCoeffs=True)-lorentz(xx))
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error on Rational Interpolation')
plt.show()

plt.clf()
print("")
print("Coefficients using pinv:")
plt.plot(xx,ratfit(x,y,n,m,xx,printCoeffs=True,pinv=True)-lorentz(xx))
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error on Rational Interpolation with pinv')
plt.show()
