import numpy as np

x_called=[]
x_called_set=set()

naiveCalls=0

def integrate_adaptive(fun,a,b,tol,extra=None):


	x=np.linspace(a,b,5)

	y=np.zeros_like(x)

	y[1]=fun(x[1])
	y[3]=fun(x[3])

	x_called.append(x[1])
	x_called.append(x[3])

	x_called_set.add(x[1])
	x_called_set.add(x[3])
	
	global naiveCalls
	naiveCalls=naiveCalls+5	

	if extra is None:
		y[0]=fun(x[0])
		y[2]=fun(x[2]) 
		y[4]=fun(x[4])

		x_called.append(x[0])
		x_called.append(x[4])
		x_called.append(x[2])

		x_called_set.add(x[0])
		x_called_set.add(x[4])
		x_called_set.add(x[2])

	else:
		y[0],y[2],y[4]=extra

	dx=x[1]-x[0]

	i1=(2*dx)*(y[0]+4*y[2]+y[4])/3
	i2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3

	if np.abs(i1-i2)<tol:

		return i2

	else:

		mid=(a+b)/2

		left=integrate_adaptive(fun,a,mid,tol/2,extra=[y[0],y[1],y[2]])
		right=integrate_adaptive(fun,mid,b,tol/2,extra=[y[2],y[3],y[4]])

		return left+right
	
y=integrate_adaptive(np.exp,-2,3,1e-10)
print("Integral is: ",y," and error is: ",y-(np.exp(3)-np.exp(-2)))
print()
if len(x_called)==len(x_called_set):
	print("No duplicates found!")

else:
	print("Duplicate found!")

print()
print("Number of function calls (optimized): ",len(x_called))
print("Number of function calls (naive): ",naiveCalls)
