import numpy as np
import matplotlib.pyplot as plt

dat=np.loadtxt('../data/lakeshore.txt')

def findCoeffs(Vl,Vr,values):

	A=np.zeros((4,4))

	A[0]=[(Vl)**k for k in range(4)]
	A[1]=[(Vr)**k for k in range(4)]

	A[2]=[k*(Vl)**(k-1) for k in range(4)]
	A[3]=[k*(Vr)**(k-1) for k in range(4)]

	Ainv=np.linalg.inv(A)

	coeffs=Ainv@values

	return coeffs

def evaluate(V,data):
	
	try:
		left=np.where(data[:,1]<=V)[0][0]
		right=np.where(data[:,1]>=V)[0][-1]
	except IndexError:
		print("Value of V is outside the allowed range")
		return None
	if left==right:
		return (data[left,0],0)
	
	else:
		
		V_left=data[left,1]
		V_right=data[right,1]

		T_left=data[left,0]
		T_right=data[right,0]

		dT_dVL=1/data[left,2]
		dT_dVR=1/data[right,2]

		values=np.array([T_left,T_right,dT_dVL,dT_dVR])

		coeffs=findCoeffs(V_left,V_right,values)
		
		T=np.sum([coeffs[k]*V**k for k in range(4)])

		Error=(T_right-T_left)/(V_right-V_left) * (V-V_left) +T_left - T

		return (T,Error) 

def lakeshore(V,data):
	
	y=0.0*V
	E=0.0*V

	if type(y)==float:
		
		y,E=evaluate(V,data)

	else:

		for k in range(len(V)):

			y[k], E[k]=evaluate(V[k],data)
			

	return y,E

xx=np.linspace(0.1,1.64,1001)

plt.plot(xx,lakeshore(xx,dat)[0],label="Interpolation")
plt.xlabel("$V$")
plt.ylabel("$T$")
plt.plot(dat[:,1],dat[:,0],'.',label="Data")
plt.title("Interpolation of $T(V)$")
plt.legend()
#plt.show()
plt.clf()

plt.plot(xx,np.abs(lakeshore(xx,dat)[1])/lakeshore(xx,dat)[0])
plt.yscale("log")
plt.ylabel('$|E/T|$')
plt.xlabel('$V$')
plt.title("Relative error estimate")
#plt.show()
