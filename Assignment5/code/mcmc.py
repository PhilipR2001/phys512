import numpy as np
import matplotlib.pyplot as plt
import camb

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

#cov=np.loadtxt("Newton_Cov.txt",delimiter=' ')
cov=np.loadtxt("Newton_Cov_constrained.txt",delimiter=' ')
planck=np.loadtxt("../data/COM_PowerSpect_CMB-TT-full_R3.01.txt",skiprows=1)
p0=np.loadtxt("planck_fit_params.txt",delimiter=' ')[:,0]
p0=np.array([69,0.022,0.12,0.06,2.1e-9,0.95])
print(p0)

errs=np.mean(planck[:,2:4],axis=1)


def getChi2(pos):
	pred=get_spectrum(pos)[:len(planck[:,0])]
	return np.sum((planck[:,1]-pred)**2/errs**2)

nsteps=5000

chain=np.zeros([nsteps,p0.size+1])

curChisq=getChi2(p0)
chain[0,0]=curChisq
chain[0,1:]=p0


curPos=p0.copy()

accepted=0

for i in range(1,nsteps):
	trialStep=np.random.multivariate_normal(0*curPos,cov)
	newPos=curPos+trialStep

	newChisq=getChi2(newPos)

	if newChisq < curChisq:

		accept=True

	else:

		dChisq=newChisq-curChisq

		if np.random.rand()<np.exp(-0.5*dChisq):
			accept=True

		
		else:
			accept=False
	
	if newPos[3]<0.054-0.0074 or newPos[3]>0.054+0.0074:
		accept=False

	if accept:
		curPos=newPos
		curChisq=newChisq
		accepted=accepted+1

	chain[i,1:]=curPos
	chain[i,0]=curChisq

np.savetxt("planck_chain_tauprior.txt",chain)
#np.savetxt("planck_chain.txt",chain)
print(accepted/nsteps)
