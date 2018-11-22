import numpy as np

def calculate(lkp, lkn, lk2, nc, n2, times, m0):
	"""The second-order self-consistent solution for the first moment:
	See Cohen paper 2, equation 54"""

	# Convert rate constants into linear space
	kp = 10**lkp
	kn = 10**lkn
	k2 = 10**lk2

	# Lamda describes rate of growth from primary nucleation
	lamda = np.sqrt(2*kp*kn*(m0**nc))
	# Kappa describes rate of growth from secondary nucleation
	kappa = np.sqrt(2*kp*k2*(m0**(n2 + 1)))
	# Number of seeds
	P0 = 0
	M0 = 0
	# Mass limit at plateau
	Minf = m0 + M0

	# These are a bunch of intermediate parameters with not much meaning
	c1 = P0*kp/kappa
	c2 = M0/(2*m0) + 0.5*(lamda/kappa)**2

	Cp = c1 + c2
	Cm = c1 - c2

	kinf = kappa*np.sqrt(2/(n2*(n2+1)) + (2*lamda**2)/(nc*kappa**2) + (2*M0)/(n2*m0) + (2*kp*P0/kappa)**2)
	kinftilde = np.sqrt(kinf**2- 4*Cp*Cm*kappa**2)

	Bp = 0.5*(kinf + kinftilde)/kappa
	Bm = 0.5*(kinf - kinftilde)/kappa

	f1 = (1 - M0/Minf)
	f2 = (Bp + Cp)/(Bm+Cp)
	f3 = (Bm * np.exp(-kappa * times) + Cp) / (Bp*np.exp(-kappa*times) + Cp)
	expon = (kinf**2)/(kappa*kinftilde)

	# Finally, calculate the aggregate mass (as a fraction of the final mass)
	Ms_time = 1 - f1*np.power(f2*f3,expon)*np.exp(-kinf*times)
	return Ms_time


