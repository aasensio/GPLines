import numpy as np
import matplotlib.pyplot as pl
import milne
import cholesky
import numpy.core.umath_tests
import scipy.optimize as opt
import scipy.linalg
from DIRECT import solve
import pdb

class milneGP(object):
	
	"""
	"""
	
#********************************
# Initialization
#********************************
	def __init__(self, xInput, yInput, noiseLevel, lineInfo, mu, whichToInvert=[True,True,False,False]):
		self.whichToInvert = np.asarray(whichToInvert)
		self.wavelength = xInput
		self.stokes = yInput
		self.noiseLevel = noiseLevel
		self.nCovariance = 0
		self.nTotalParCovariance = 0
		self.nParCovariance = []		
		self.funCovariance = []
		self.lineInfo = lineInfo
		self.synth = milne.milne(self.lineInfo)
		self.nTotalPars = 8
		self.nParsModel = 8
		self.nWavelengths = self.synth.lineInfo[-1]
		self.nWavelengthsTotal = len(self.stokes)
		self.parsToInvert = np.where(self.whichToInvert == True)
		self.nParsToInvert = np.sum(self.whichToInvert)
		self.maxLike = 1e10
		self.mu = mu
		
		self.covAddTypes = {'sqr' : self.addCovarianceSquareExponential, 'periodic' : self.addCovariancePeriodic}
		self.covTypes = {'sqr' : self.covarianceSquareExponential, 'periodic' : self.covariancePeriodic}
		return
	
#********************************
# Addition of covariance functions
#********************************
	def addCovariance(self, whichType):
		"""
		Add covariance function to the full list of covariances
		"""		
		self.covAddTypes[whichType]()
		return
	
#********************************
# Addition of square exponential covariance function
#********************************
	def addCovarianceSquareExponential(self):
		"""
		Add a square exponential covariance matrix
		"""
		self.nCovariance += 1
		self.nParCovariance.append(2)
		self.nTotalParCovariance += 2
		self.nTotalPars += 2
		self.funCovariance.append('sqr')
		return
	
#********************************
# Addition of periodic covariance function
#********************************
	def addCovariancePeriodic(self):
		"""
		Add a square exponential covariance matrix
		"""
		self.nCovariance += 1
		self.nParCovariance.append(2)
		self.nTotalParCovariance += 2
		self.nTotalPars += 2
		self.funCovariance.append('periodic')
		return
	
#********************************
# Compute full covariance function
#********************************
	def covariance(self, covPars, xLeft, xRight):
		"""
		Covariance matrix. It returns the value of the covariance matrix
		and the derivative of the matrix wrt to the hyperparameters iterating
		over all defined covariances
		"""
		
# Broadcast to build the covariance function in one step
		nLeft = len(xLeft)
		nRight = len(xRight)
		xLeftB = xLeft[np.newaxis,:]
		xRightB = xRight[:,np.newaxis]
		
		left = 0
		KOut = np.zeros((nLeft,nRight))
		dKOut = np.zeros((nLeft,nRight,self.nTotalParCovariance))
		for i in range(self.nCovariance):
			right = left + self.nParCovariance[i]
			pars = covPars[left:right]
			K, dK = self.covTypes[self.funCovariance[i]](pars, xLeftB, xRightB)
			KOut += K
			dKOut[:,:,left:right] = dK
			left = right
		return KOut, dKOut
		
#********************************
# Return square exponential covariance function
#********************************
	def covarianceSquareExponential(self, pars, xLeft, xRight):
		"""
		Covariance matrix. It returns the value of the covariance matrix
		and the derivative of the matrix wrt to the hyperparameters
		"""
		
		x = xLeft - xRight
		lambdaGP = pars[0]
		sigmaGP = pars[1]
	
		expon = np.exp(-0.5 * lambdaGP * x**2)
	
# Covariance matrix
		K = sigmaGP * expon
	
# Derivatives of the covariance matrix
		dKdsigmaGP = expon
		dKdlambdaGP = -0.5 * K * x**2
	
		return K, np.dstack((dKdlambdaGP, dKdsigmaGP))
	
#********************************
# Return periodic covariance function
#********************************
	def covariancePeriodic(self, pars, xLeft, xRight):
		"""
		Covariance matrix. It returns the value of the covariance matrix
		and the derivative of the matrix wrt to the hyperparameters
		"""
		
		x = xLeft - xRight
		lambdaGP = pars[0]
		sigmaGP = pars[1]
	
		expon = np.exp(-0.5 * lambdaGP * x**2)
	
# Covariance matrix
		K = sigmaGP * expon
	
# Derivatives of the covariance matrix
		dKdsigmaGP = expon
		dKdlambdaGP = -0.5 * K * x**2
	
		return K, np.dstack((dKdlambdaGP, dKdsigmaGP))
	
#********************************
# Compute marginal likelihood for the GP
#********************************
	def marginalLikelihood(self, pars):
		"""
		"""
		
		covPars = np.exp(pars[0:self.nTotalParCovariance])
		funPars = pars[self.nTotalParCovariance:]
		
		self.K, self.dK = self.covariance(covPars, self.wavelength, self.wavelength)

# Compute covariance matrix and invert it
		self.C = self.K + self.noiseLevel**2 * np.identity(self.nWavelengths)
		
# Invert each block and then build the full matrix
		self.CInv, logD = cholesky.cholInvert(self.C)
				
# Call the model for the mean				
		w, self.S, dS = self.synth.synthDerivatives(funPars, self.mu)
		
		likelihood = 0.0
		jacobian = np.zeros(self.nTotalPars)
		
# Put all Stokes parameters in a 1D vector
		for indStokes in self.parsToInvert[0]:
			
			synthStokes = self.S[indStokes,:]
			synthStokesJac = dS[:,indStokes,:]
				
			residual = self.stokes[indStokes] - synthStokes
		
			likelihood += 0.5 * np.dot(np.dot(residual.T,self.CInv),residual) + 0.5 * logD
		
			alpha = np.dot(self.CInv, residual)
		
# Computation of the Jacobian
# First for the parameters of the covariance function		
			for i in range(self.nTotalParCovariance):
				jacobian[i] += -0.5 * np.sum(numpy.core.umath_tests.inner1d(self.CInv, self.dK[:,:,i].T)) + 0.5*np.dot(np.dot(alpha.T,self.dK[:,:,i]),alpha)

# Chain rule to take into account that we use the exponential of the parameters			
			jacobian[0:self.nTotalParCovariance] *= -covPars
				
# And then the Jacobian of the mean
			for i in range(self.nParsModel):
				jacobian[self.nTotalParCovariance+i] = -np.dot(alpha.T,synthStokesJac[i,:])
		
		if (likelihood < self.maxLike):
			self.maxLike = likelihood
			print self.maxLike, np.log(covPars)
			
		return likelihood, jacobian
	
#********************************
# Calling routine for DIRECT
#********************************
	def funcWrapperDIRECT(self, x, user_data):
		f, j = self.marginalLikelihood(x)
		return f, 0
	
#********************************
# Optimize the GP with DIRECT
#********************************
	def optimizeGPDIRECT(self):
		"""
		Optimize the marginal posterior of the GP to obtain the covariance and model parameters
		"""
		
# Do a sensible initialization				
		lower = np.zeros(self.nTotalPars)
		upper = np.zeros(self.nTotalPars)
		lower[0:self.nTotalParCovariance] = [np.log(1.0/3.0**2),np.log(1e-5**2)]
		upper[0:self.nTotalParCovariance] = [np.log(1.0/0.5**2),np.log(1e-2**2)]
		lower[self.nTotalParCovariance:] = [0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.03, 0.0]
		upper[self.nTotalParCovariance:] = [2000.0, 180.0, 180.0, 5.0, 2.0, 4.0, 0.3, 10.0]
		x, fmin, ierror = solve(self.funcWrapperDIRECT, lower, upper, volper=1e-15, algmethod=1)
		self.optimalPars = x
		self.marginalLikelihood(self.optimalPars)
		return
	
#********************************
# Optimize the GP
#********************************
	def optimizeGP(self):
		"""
		Optimize the marginal posterior of the GP to obtain the covariance and model parameters
		"""
		
# Do a sensible initialization		
		x0 = self.optimalPars
		bnds = [(np.log(1.0/3.0**2),np.log(1.0/0.5**2)),(np.log(1e-5**2),np.log(1e-2**2))]
		bnds[self.nTotalParCovariance:] = [(0,2000.0), (0.0, 180.0), (0.0, 180.0), (-5.0, 10.0), (0.0, 2.0), (0.0, 4.0), (0.03,0.3), (0.0, 10.0)]
		res = opt.minimize(self.marginalLikelihood, x0, method='L-BFGS-B', jac=True, bounds=bnds)
		self.optimalPars = res.x
		self.marginalLikelihood(self.optimalPars)
	
#********************************
# Prediction of the GP including and not including the mean
#********************************
	def predictGP(self, pars):
		
		covPars = np.exp(pars[0:self.nTotalParCovariance])
		funPars = pars[self.nTotalParCovariance:]
		
		w, S = self.synth.synth(funPars, self.mu)
		
		K, dK = self.covariance(covPars, self.wavelength, self.wavelength)
		C = K + self.noiseLevel**2 * np.identity(self.nWavelengths)
		
		CInv, logD = cholesky.cholInvert(C)
		
		KStar, dKStar = self.covariance(covPars, self.wavelength, self.wavelength)
		
		outStokes = np.zeros((4,self.nWavelengths))
		
		for i in range(4):
			if (self.whichToInvert[i] == True):
				outStokes[i,:] = np.dot(KStar,np.dot(CInv, (self.stokes[i,:] - S[i,:]))) + S[i,:]
		
		return S, outStokes