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

planck=np.loadtxt("../data/COM_PowerSpect_CMB-TT-full_R3.01.txt",skiprows=1)
errs=np.mean(planck[:,2:4],axis=1)

chain=np.loadtxt("planck_chain_tauprior.txt",delimiter=' ')

def getChi2(pos):
	pred=get_spectrum(pos)[:len(planck[:,0])]
	return np.sum((planck[:,1]-pred)**2/errs**2)

p=np.zeros(6)
p_errs=p.copy()

for i in range(len(p)):
	p[i]=np.mean(chain[500:,i+1])
	p_errs[i]=np.std(chain[500:,i+1])

print(p)
print(p_errs)


print(getChi2(p))

planck=np.loadtxt("../data/COM_PowerSpect_CMB-TT-full_R3.01.txt",skiprows=1)

plt.plot(planck[:,0],planck[:,1],'.',label='Data')
plt.plot(planck[:,0],get_spectrum(p)[:len(planck[:,0])],label='Fit')
plt.xlabel('Multipole')
plt.ylabel('Variance')
plt.legend()
plt.title('Constrained MCMC Fit')
plt.show()



