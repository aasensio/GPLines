import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
from andres import voigt, cholInvert, voigtZeeman
import brewer2mpl 
import pdb
import numpy.core.umath_tests

# Function that returns a line in absorption as f=1-d*H(v/delta,a)
# with H(v,a) the Voigt function. If jacobian=True, it also returns
# the derivatives wrt to the parameter
# The parameters are [d, delta, a]
def funcVoigt(x, pars, jacobian=False):
	d, width, a = pars
	v = x / width
	H, L = voigtZeeman(v, a)
	out = 1.0 - d*H
	jac = np.zeros((3,x.size))
	jac[0,:] = -H
	jac[1,:] = d * (-2.0*v*H + 2.0*a*L) * (x / width**2)
	jac[2,:] = -d * (-2.0 / np.sqrt(np.pi) + 2.0*a*H + 2.0*v*L)
	if (jacobian == False):
		return out
	else:
		return out, jac

# Wrapper of the line profile for the LS inversion
def func(x, d, width, damping):
	if (d < 0 or width < 0 or damping < 0):
		return 1e10
	else:
		return funcVoigt(x, [d, width, damping], jacobian=False)

# Covariance matrix. It returns the value of the covariance matrix
# and the derivative of the matrix wrt to the hyperparameters
def covariance(x, pars):
	lambdaGP = pars[0]
	sigmaGP = pars[1]
	
	expon = np.exp(-0.5 * lambdaGP * x**2)
	
# Covariance matrix
	K = sigmaGP * expon
	
# Derivatives of the covariance matrix
	dKdsigmaGP = expon
	dKdlambdaGP = -0.5 * K * x**2
	
	return K, dKdlambdaGP, dKdsigmaGP

# Returns the marginal likelihood for a Gaussian process
def marginalLikelihoodVoigt(pars, *args):
	xInput = args[0]
	yInput = args[1]
	sigmaNoise = args[2]
		
	lambdaGP = np.exp(pars[0])
	sigmaGP = np.exp(pars[1])
	
	modelPars = pars[2:]
					
	K, dKdl, dKds = covariance(xInput[np.newaxis,:]-xInput[:,np.newaxis], [lambdaGP, sigmaGP])
			
	C = K + sigmaNoise**2 * np.identity(len(xInput))

	CInv, logD = cholInvert(C)
	
	model, jacVoigt = funcVoigt(xInput, modelPars, jacobian=True)
	residual = yInput - model
	likelihood = 0.5 * np.dot(np.dot(residual.T,CInv),residual) + 0.5 * logD
	
# Jacobian
	jac = np.zeros(5)
	
# dLdlambda
	residual2 = np.dot(CInv, residual)
	jac[0] = -0.5 * np.sum(numpy.core.umath_tests.inner1d(CInv, dKdl.T)) + 0.5*np.dot(np.dot(residual2.T,dKdl),residual2)
	jac[0] = -jac[0] * lambdaGP
	
# dLdsigma
	jac[1] = -0.5 * np.sum(numpy.core.umath_tests.inner1d(CInv, dKds.T)) + 0.5*np.dot(np.dot(residual2.T,dKds),residual2)
	jac[1] = -jac[1] * sigmaGP
	
	for i in range(3):
		jac[i+2] = -np.dot(residual2.T,jacVoigt[i,:])
			
	return likelihood, jac

# Example with a Voigt function fitted with a Gaussian
def VoigtExample():
	nPoints = 50
	sigmaNoise = 0.002

	dampingOrig = 0.5
	widthOrig = 1.0
	dOrig = 0.5

	x = np.linspace(-5.0,5.0,nPoints)
	y = funcVoigt(x, [dOrig, widthOrig, dampingOrig]) + 0.001*x**2 - 0.01*x + 0.1
	ynoise = y + sigmaNoise * np.random.randn(nPoints) 

	args = [x, ynoise, sigmaNoise]

	# Initial conditions and optimize the merit function
	x0 = [1.0, 1.0, 0.5, 1.0, 0.2] #, 0.0, 0.0]
	
	res = opt.minimize(marginalLikelihoodVoigt, x0, method='BFGS', jac=True, args=args)
	fit, cov = opt.curve_fit(func, x, ynoise)

	lambdaGP, sigmaGP, d, width, damping = res.x
	lambdaGP = np.exp(lambdaGP)
	sigmaGP = np.exp(sigmaGP)

	# Compute the covariance matrix
	K, dKdl, dKds = covariance(x[np.newaxis,:]-x[:,np.newaxis], [lambdaGP, sigmaGP])

	C = K + sigmaNoise**2 * np.identity(nPoints)

	CInv, logD = cholInvert(C)

	# Predictive distribution
	xStar = np.linspace(-5.0, 5.0,nPoints)
	predicted = np.zeros(nPoints)

	for i in range(nPoints):
		m = funcVoigt(x, [d, width, damping])
		mStar = funcVoigt(xStar[i], [d, width, damping])
		
		KStar = sigmaGP * np.exp(-0.5 * lambdaGP * (xStar[i]-x)**2)
		predicted[i] = np.dot(KStar,np.dot(CInv, (ynoise-m))) + mStar

	print "d_GP={0} - w_GP={1} - a_GP={2}".format(d,width,damping)
	print "d_LS={0} - w_LS={1} - a_LS={2}".format(fit[0],fit[1],fit[2])

	bmap = brewer2mpl.get_map('dark2', 'Qualitative', 6).mpl_colors

	bestFit = funcVoigt(x, [d, width, damping])
	LSFit = funcVoigt(x, fit)
	OriginalVoigt = funcVoigt(x, [dOrig, widthOrig, dampingOrig])
	
	pl.close('all')
	fig = pl.figure(num=1, figsize=(15,8))
	ax = fig.add_subplot(1,2,1)
	ax.plot(x, ynoise, label='Noisy original', color=bmap[0], linewidth=3)
	ax.plot(xStar, predicted, label='Prediction', color=bmap[1], linewidth=3)
	ax.plot(x, bestFit, label='Best fit', color=bmap[2], linewidth=3)
	ax.plot(x, y, label='Orig.', color=bmap[3], linewidth=3)
	ax.plot(x, LSFit, '--', label='LS', color=bmap[4], linewidth=3)
	ax.plot(x, OriginalVoigt, '--', label='Orig. Voigt', color=bmap[5], linewidth=3)
	ax.legend(loc='lower left')

	ax = fig.add_subplot(1,2,2)
	ax.plot(x, ynoise - bestFit, label='Residual GP fit', color=bmap[0], linewidth=3)
	ax.plot(x, y - OriginalVoigt, label='Original systematics', color=bmap[1], linewidth=3)
	ax.legend(loc='upper right')
	
	fig.savefig("example.pdf")