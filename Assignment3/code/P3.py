import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("../data/dish_zenith.txt")

x=data[:,0]
y=data[:,1]
z=data[:,2]

A=np.zeros([len(x),4])


A[:,0]=x**2+y**2
A[:,1]=-2*x
A[:,2]=-2*y
A[:,3]=x**0


lhs=np.transpose(A)@A
rhs=np.transpose(A)

m=np.linalg.inv(lhs)@rhs@z


a=m[0]
x0=m[1]/a
y0=m[2]/a
z0=m[3]-a*(x0**2+y0**2)

print("a=",a,", x0=",x0,", y0=",y0,", z0=",z0)



pred=z0+a*((x-x0)**2+(y-y0)**2)

N=np.mean((pred-z)**2)

print("N=",N)


dA=np.sqrt(N*np.diag(np.linalg.inv(np.transpose(A)@A)))

print(dA)

f=1/(4*a)
df=dA[0]/(4*a**2)

print("f=",f/1000,"+/-",df/1000,"m")
