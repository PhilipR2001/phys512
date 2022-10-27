import numpy as np
import camb
import matplotlib.pyplot as plt


def get_spectrum(pars,lmax=3000):
    
	H0=pars[0]
	ombh2=pars[1]
	omch2=pars[2]
	tau=pars[3]
	As=pars[4]
	ns=pars[5]
	pars=camb.CAMBparams()
	pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
	pars.InitPower.set_params(As=As,ns=ns,r=0)
	pars.set_for_lmax(lmax,lens_potential_accuracy=0)
	results=camb.get_results(pars)
	powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
	cmb=powers['total']
	tt=cmb[:,0]
	return tt[2:]


def deriv(powerf,x,p):
	
	grad=np.zeros([x.size,p.size])

	dp=1e-8


	for i in range(p.size):

		pright=p.copy()
		pright[i]=p[i]+dp

		pleft=p.copy()
		pleft[i]=p[i]-dp

		dfdp=(powerf(pright)[:len(x)]-powerf(pleft)[:len(x)])/(2*dp)

		grad[:,i]=dfdp

	return grad

planck=np.loadtxt("../data/COM_PowerSpect_CMB-TT-full_R3.01.txt",skiprows=1)
pars=np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
model=get_spectrum(pars)[:len(planck[:,0])]


errs=np.mean(planck[:,2:4],axis=1)
print(len(errs))
N=np.diag((errs)**2)
chi2=np.sum((model-planck[:,1])**2/(errs**2))

print("Chi^2=",chi2)


### PART 2 ###

p=pars.copy()
l=0.8
for i in range(15):
	print(i)	
	pred=get_spectrum(p)[:len(planck[:,0])]
	grad=deriv(get_spectrum,planck[:,0],p)
	r=(planck[:,1]-pred)
	lhs=grad.T@np.diag((np.diag(N))**(-1))@grad
	lhs=lhs+l*np.diag(np.diag(lhs))
	rhs=grad.T@np.diag((np.diag(N))**(-1))@r
	
	u,s,v=np.linalg.svd(lhs)
	lhsinv=v.T@np.diag(s**-1)@u.T

	dp=lhsinv@rhs
	for j in range(len(p)):

		p[j]=p[j]+dp[j]

		if j==3 and (p[j]<0.054-0.0074 or p[j]>0.054+0.0074):
			p[j]=p[j]-dp[j]		
			
	#print(p)
	#print(np.sum((r**2)/errs**2))
newpred=get_spectrum(p)[:len(planck[:,0])]

plt.plot(planck[:,0],planck[:,1],'.')
plt.plot(planck[:,0],newpred,'.')
plt.show()
print(p)

errparams=np.sqrt(np.diag(lhsinv))
#np.savetxt("planck_fit_params.txt",np.transpose([p,errparams]))
#np.savetxt("Newton_Cov.txt",lhsinv)

np.savetxt("planck_fit_params_constrained.txt",np.transpose([p,errparams]))
np.savetxt("Newton_Cov_constrained.txt",lhsinv)
print("errors=",errparams)
print("New Chi^2=",np.sum((newpred-planck[:,1])**2/(errs**2)))


