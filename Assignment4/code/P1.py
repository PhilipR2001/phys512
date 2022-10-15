import numpy as np
import matplotlib.pyplot as plt


data=np.load('../data/sidebands.npz')

t=data['time']
d=data['signal']


def calc_lorentzian(t,p):

	y=p[0]/(1+((t-p[1])/p[2])**2)

	grad=np.zeros([t.size,p.size])

	grad[:,0]=1/(1+((t-p[1])/p[2])**2)
	grad[:,1]=(2*p[0]*((t-p[1])/(p[2])**2))/(1+((t-p[1])/p[2])**2)**2
	grad[:,2]=(2*p[0]*(t-p[1])**2/((p[2])**3))/(1+((t-p[1])/p[2])**2)**2

	return y, grad


### PART A ###

p=np.array([1.5,0.000200,0.000025])

guess,gradguess=calc_lorentzian(t,p)

for i in range(10):
	
	
	pred,grad=calc_lorentzian(t,p)
	
	r=(d-pred)
	
	lhs=grad.T@grad
	rhs=grad.T@r

	dp=np.linalg.inv(lhs)@rhs

	for j in range(p.size):
		p[j]=p[j]+dp[j]


print("Best fit parameters: ",p)

fit=calc_lorentzian(t,p)[0]

plt.plot(t,d,'.',label='Data')
plt.plot(t,guess,label='Initial Guess')
plt.plot(t,fit,color='k',label='Best Fit')

plt.xlabel('Time')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.ylabel('Signal')
plt.title('Lorentzian Fit')
plt.legend()
plt.show()



#### PART B ###

N=np.mean((d-pred)**2)

print("N=",(N))

A=(calc_lorentzian(t,p)[1])

errs=np.sqrt(np.diag(np.linalg.inv(N**(-1)*A.T@A)))

print("Error on parameters: ",errs)

print("Chi^2=",1/N*np.sum((d-pred)**2))



### PART C ###

def lorentzian(t,p):

	return p[0]/(1+(t-p[1])**2/(p[2])**2)



def deriv(fun,t,p):
	

	grad=np.zeros([t.size,p.size])

	dp=1e-5

	for i in range(p.size):

		pright=p.copy()
		pright[i]=p[i]+dp

		pleft=p.copy()
		pleft[i]=p[i]-dp

		dfdp=(fun(t,pright)-fun(t,pleft))/(2*dp)

		grad[:,i]=dfdp


	return grad


p=np.array([1.5,0.000200,0.000025])

guess=lorentzian(t,p)

for i in range(10):
	
	
	pred=lorentzian(t,p)
	grad=deriv(lorentzian,t,p)

	r=(d-pred)
	
	lhs=grad.T@grad
	rhs=grad.T@r

	dp=np.linalg.inv(lhs)@rhs

	for j in range(p.size):
		p[j]=p[j]+dp[j]


fitnum=lorentzian(t,p)
#plt.plot(t,d)
#plt.plot(t,guess)
#plt.plot(t,fitnum)
#plt.show()

print()
print("Using Numerical Derivatives:")
print("Best Fit Parameters:",p)

A=deriv(lorentzian,t,p)

errs=np.sqrt(np.diag(np.linalg.inv(N**(-1)*A.T@A)))

print("Error on parameters: ",errs)

print("Chi^2=",1/N*np.sum((d-pred)**2))




### PART D ###

# p has the form [a,t0,w,b,c,dt]

def threeLorentzian(t,p):
	
	return lorentzian(t,p[:3]) + p[3]/(1+(t-p[1]+p[5])**2/(p[2])**2)+ p[4]/(1+(t-p[1]-p[5])**2/(p[2])**2)



p=np.array([1.423,1.923e-4,1.971e-5,0.1,0.1,0.000050])

guess=threeLorentzian(t,p)


for i in range(10):
	
	
	pred=threeLorentzian(t,p)
	grad=deriv(threeLorentzian,t,p)

	r=(d-pred)
	
	lhs=grad.T@grad
	rhs=grad.T@r

	dp=np.linalg.inv(lhs)@rhs

	for j in range(p.size):
		p[j]=p[j]+dp[j]


fitThree=threeLorentzian(t,p)
plt.plot(t,d,'.',label='Data')
plt.plot(t,guess,label='Initial Guess')
plt.plot(t,fitThree,label='Best Fit',color='k')
plt.xlabel('Time')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.ylabel('Signal')
plt.title('Triple Lorentzian Fit')
plt.legend()
plt.show()

print()
print("Best Fit Parameters:",p)

A=deriv(threeLorentzian,t,p)

errs=np.sqrt(np.diag(np.linalg.inv(N**(-1)*A.T@A)))

print("Error on Parameters:",errs)

print("Chi^2=",N**(-1)*np.sum((fitThree-d)**2))




### PART E ###
plt.plot(t,fitThree-d)
plt.xlabel('Time')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.ylabel('Residuals')
plt.title('Residuals on Triple Lorentzian Fit')


plt.show()



### PART F ###

cov=N*np.linalg.inv(A.T@A)

L=np.linalg.cholesky(cov)
pPert=p+L@np.random.randn(6)

fitPert=threeLorentzian(t,pPert)

plt.plot(t,d,'.',label='Data')
plt.plot(t,threeLorentzian(t,pPert),label='Perturbed Best Fit',color='k')
plt.xlabel('Time')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.ylabel('Signal')
plt.title('Perturbed Triple Lorentzian Fit')
plt.legend()
plt.show()


plt.plot(t,fitPert-d,label='Perturbed')
plt.plot(t,fitThree-d,label='Non-Perturbed')
plt.xlabel('Time')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.ylabel('Residuals')
plt.title('Residuals on Best Fit and Perturbed Fit')
plt.legend()
plt.show()

print("New Chi^2=",N**(-1)*np.sum((fitPert-d)**2))



### PART G ###


p=np.array([1.423,1.923e-4,1.971e-5,0.1,0.1,0.000050])

def getChiSq(pos):
	pred=threeLorentzian(t,pos)
	return 1/N*(d-pred).T@(d-pred)

nsteps=20000
nparams=p.size

params=np.zeros([nsteps,nparams+1])

params[0,0:-1]=p


curChisq=getChiSq(p)

params[0,-1]=curChisq


curPos=p.copy()

accepted=0
for i in range(1,nsteps):

	trialStep=np.random.multivariate_normal(0*curPos,cov)

	newPos=curPos+trialStep

	newChisq=getChiSq(newPos)

	if newChisq<curChisq:
		accept=True

	else:

		dChisq=newChisq-curChisq

		if np.random.rand()<np.exp(-0.5*dChisq):
			accept=True
		
		else:
			accept=False

	if accept:
		curPos=newPos
		curChisq=newChisq
		accepted=accepted+1

	params[i,0:-1]=curPos
	params[i,-1]=curChisq

print()
print('MCMC:')
mcmcParams=np.mean(params[2500:,0:-1],axis=0)
mcmcErrors=np.std(params[2500:,0:-1],axis=0)
print('Parameters:',mcmcParams)
print('Errors:',mcmcErrors)

print('Chi^2=',getChiSq(mcmcParams))
plt.plot(t,d,'.',label='Data')
plt.plot(t,threeLorentzian(t,mcmcParams),label='MCMC Fit',color='k')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
plt.title('MCMC Fit')
plt.legend()
plt.show()

for i in range(1):
	plt.plot((params[:,i]))
	plt.xlabel('Steps')
	plt.ylabel('a')
	plt.title('Evolution of the first parameter during MCMC')
	plt.show()
print('Acceptance ratio=',accepted/nsteps)
