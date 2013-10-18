import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
from andres import voigt, cholInvert
import brewer2mpl 
import pdb

def funcLinear(x, pars):
	a, b = pars
	return a*x + b

def funcParabola(x, pars):
	a, b, c = pars
	return a*x**2 + b*x + c

def funcGaussian(x, pars):
	d, width = pars	
	return 1.0 - d*voigt(x / width, 0.0)

def funcVoigt(x, pars):
	d, width, damping = pars	
	return 1.0 - d*voigt(x / width, damping)

def func(x, d, width, damping):
	if (d < 0 or width < 0 or damping < 0):
		print d, width, damping
		return 1e10
	else:
		return funcVoigt(x, [d, width, damping])

def func2(x, a, b):
	return funcLinear(x, [a, b])
	
def marginalLikelihoodVoigt(pars, *args):	
	xInput = args[0]
	yInput = args[1]
	sigmaNoise = args[2]
		
	lambdaGP = np.exp(pars[0])
	sigmaGP = np.exp(pars[1])
	
	modelPars = pars[2:]
					
	K = sigmaGP * np.exp(-0.5 * lambdaGP * (xInput[np.newaxis,:]-xInput[:,np.newaxis])**2)
			
	C = K + sigmaNoise**2 * np.identity(len(xInput))

	CInv, logD = cholInvert(C)
	
	model = funcVoigt(xInput, modelPars)
	residual = yInput - model
	likelihood = 0.5 * np.dot(np.dot(residual.T,CInv),residual) + 0.5 * logD
	print 'L=', likelihood, ' - l=', lambdaGP, ' - s=', sigmaGP
	return likelihood

def marginalLikelihoodLinear(pars, *args):	
	xInput = args[0]
	yInput = args[1]
	sigmaNoise = args[2]
		
	lambdaGP = np.exp(pars[0])
	sigmaGP = np.exp(pars[1])
	
	modelPars = pars[2:]
					
	K = sigmaGP * np.exp(-0.5 * lambdaGP * (xInput[np.newaxis,:]-xInput[:,np.newaxis])**2)
			
	C = K + sigmaNoise**2 * np.identity(len(xInput))

	CInv, logD = cholInvert(C)
	
	model = funcLinear(xInput, modelPars)
	residual = yInput - model
	likelihood = 0.5 * np.dot(np.dot(residual.T,CInv),residual) + 0.5 * logD
	print 'L=', likelihood, ' - l=', lambdaGP, ' - s=', sigmaGP, modelPars
	return likelihood

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
	res = opt.minimize(marginalLikelihoodVoigt, x0, method='BFGS', jac=False, args=args)
	fit, cov = opt.curve_fit(func, x, ynoise)

	lambdaGP, sigmaGP, d, width, damping = res.x
	lambdaGP = np.exp(lambdaGP)
	sigmaGP = np.exp(sigmaGP)

	# Compute the covariance matrix
	K = sigmaGP * np.exp(-0.5 * lambdaGP * (x[np.newaxis,:]-x[:,np.newaxis])**2)

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
	
	
# Example with a parabola fitted with a linear function
def ParabolaExample():
	nPoints = 50
	sigmaNoise = 0.5

	aOrig = 0.5
	bOrig = 1.0
	cOrig = 0.5

	x = np.linspace(-5.0,5.0,nPoints)
	y = funcParabola(x, [aOrig, bOrig, cOrig])
	ynoise = y + sigmaNoise * np.random.randn(nPoints)

	args = [x, ynoise, sigmaNoise]

	# Initial conditions and optimize the merit function
	x0 = [-1.5, 0.1, 0.5, 1.0]
	res = opt.minimize(marginalLikelihoodLinear, x0, method='BFGS', jac=False, args=args)
	fit, cov = opt.curve_fit(func2, x, ynoise)

	lambdaGP, sigmaGP, a, b = res.x
	lambdaGP = np.exp(lambdaGP)
	sigmaGP = np.exp(sigmaGP)

	# Compute the covariance matrix
	K = sigmaGP * np.exp(-0.5 * lambdaGP * (x[np.newaxis,:]-x[:,np.newaxis])**2)

	C = K + sigmaNoise**2 * np.identity(nPoints)

	CInv, logD = cholInvert(C)

	# Predictive distribution
	xStar = np.linspace(-5.0, 5.0,nPoints)
	fStar = np.zeros(nPoints)

	for i in range(nPoints):
		m = funcLinear(x, [a, b])
		mStar = funcLinear(xStar[i], [a, b])
		
		KStar = sigmaGP * np.exp(-0.5 * lambdaGP * (xStar[i]-x)**2)
		fStar[i] = np.dot(KStar,np.dot(CInv, (ynoise-m))) + mStar

	print 'd_GP=', a, ' - w_GP=', b
	print 'd_LS=', fit[0], ' - w_LS=', fit[1]

	bmap = brewer2mpl.get_map('dark2', 'Qualitative', 6).mpl_colors

	pl.close('all')
	fig = pl.figure(num=1, figsize=(15,8))
	ax = fig.add_subplot(1,2,1)
	ax.plot(x, ynoise, label='Noisy original', color=bmap[0], linewidth=3)
	ax.plot(xStar, fStar, label='Prediction', color=bmap[1], linewidth=3)
	ax.plot(x, funcLinear(x, [a, b]), label='Best fit', color=bmap[2], linewidth=3)
	ax.plot(x, y, label='Orig.', color=bmap[3], linewidth=3)
	ax.plot(x, funcLinear(x, fit), '--', label='LS', color=bmap[4], linewidth=3)
	ax.plot(x, funcLinear(x, [aOrig, bOrig]), '--', label='Orig. Linear', color=bmap[5], linewidth=3)
	ax.legend(loc='lower left')

	ax = fig.add_subplot(1,2,2)
	ax.plot(x, ynoise - funcLinear(x, [a, b]))
	ax.plot(x, ynoise - funcLinear(x, fit))