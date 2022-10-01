import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


#In Gyrs

half_lives=1/3.156e16*np.array([1.41e17,2.08e6,2.4e4,7.74e15,2.38e15,5.05e13,3.3e5,186,1608,1194,1.643e-4,7.04e8,1.58e8,1.20e7])

taus=half_lives/np.log(2)


def fun(x,y,taus=taus):

	dydx=np.zeros(len(taus)+1)

	dydx[0]=-y[0]/taus[0]

	for i in range(1,len(dydx)-1):
		dydx[i]= -y[i]/taus[i] + y[i-1]/taus[i-1]

	dydx[-1]=y[-2]/taus[-1]

	return dydx


y=np.zeros(len(taus)+1)
y[0]=1
ans=si.solve_ivp(fun,[0,20],y,method="Radau")

t=np.linspace(0,20,1001)
plt.plot(ans.t,ans.y[-1,:]/ans.y[0,:],label="Numerical solution")
plt.plot(t,np.exp(t/taus[0])-1,label="Analytical approximation")
plt.xlabel("$t$ [Gyr]")
plt.ylabel("$N_{Pb206}/N_{U238}$")
plt.legend()
plt.title("Ratio of Pb206 and U238")
plt.show()


y=0*y
y[0]=1
ans=si.solve_ivp(fun,[0,2],y,method="Radau")


plt.plot(ans.t,ans.y[4,:]/ans.y[3,:])
plt.xlabel("$t$ [Gyr]")
plt.ylabel("$N_{Th230}/N_{U234}$")
plt.title("Ratio of Th230 to U234")
plt.show()
