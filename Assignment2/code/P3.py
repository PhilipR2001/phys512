import numpy as np
import matplotlib.pyplot as plt

n=40

xp=np.linspace(-1,1,n)

y=np.log2(xp+3)-2


print("First few coefficients: ",np.polynomial.chebyshev.chebfit(xp,y,n-1)[:10])

coeffs=np.polynomial.chebyshev.chebfit(xp,y,n-1)[:8]

eval=np.polynomial.chebyshev.chebval

def ln(x):

	m_e,p_e=np.frexp(np.e)
	log_2e=eval(4*m_e-3,coeffs)+p_e
	m,p=np.frexp(x)

	return (eval(4*m-3,coeffs)+p)/log_2e

x=np.linspace(0.01,4,1001)

plt.plot(x,ln(x)-np.log(x))
plt.title("Comparison between my function and np.log")
plt.xlabel("x")
plt.ylabel("Error")

plt.show()


