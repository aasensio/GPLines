import numpy as np
import matplotlib.pyplot as pl
import milne
import cholesky
import numpy.core.umath_tests

class milneGP(object):
	
	"""
	"""
	
	def __init__(self, xInput, yInput, noiseLevel, lineInfo):
		self.wavelength = xInput
		self.stokes = yInput
		self.noiseLevel = noiseLevel
		self.nCovariance = 0
		self.nTotalParCovariance = 0
		self.nParCovariance = []		
		self.funCovariance = []
		self.lineInfo = lineInfo
		self.synth = milne.milne(self.lineInfo)
		self.nTotalPars = 9
		self.nParsModel = 9
		self.nWavelengths = self.synth.lineInfo[-1]
		
		self.covAddTypes = {'sqr' : self.addCovarianceSquareExponential}
		self.covTypes = {'sqr' : self.covarianceSquareExponential}
		return
	
# Addition of covariance functions
	def addCovariance(self, whichType):
		"""
		Add covariance function to the full list of covariances
		"""		
		self.covAddTypes[whichType]()
		return
	
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
	
# Compute the covariance matrix
	def covariance(self, covPars):
		"""
		Covariance matrix. It returns the value of the covariance matrix
		and the derivative of the matrix wrt to the hyperparameters iterating
		over all defined covariances
		"""
		left = 0
		self.K = np.zeros((self.nWavelengths,self.nWavelengths))
		self.dK = np.zeros((self.nWavelengths,self.nWavelengths,self.nTotalParCovariance))
		for i in range(self.nCovariance):
			right = left + self.nParCovariance[i]
			pars = covPars[left:right]
			K, dK = self.covTypes[self.funCovariance[i]](pars)			
			self.K += K
			self.dK[:,:,left:right] = dK
			left = right
		return
		
	def covarianceSquareExponential(self, pars):
		"""
		Covariance matrix. It returns the value of the covariance matrix
		and the derivative of the matrix wrt to the hyperparameters
		"""
		
		x = self.wavelength[np.newaxis,:]-self.wavelength[:,np.newaxis]
		lambdaGP = pars[0]
		sigmaGP = pars[1]
	
		expon = np.exp(-0.5 * lambdaGP * x**2)
	
# Covariance matrix
		K = sigmaGP * expon
	
# Derivatives of the covariance matrix
		dKdsigmaGP = expon
		dKdlambdaGP = -0.5 * K * x**2
	
		return K, np.dstack((dKdlambdaGP, dKdsigmaGP))
	
	def marginalLikelihood(self, pars):
		"""
		"""
		
		covPars = np.exp(pars[0:self.nTotalParCovariance])
		funPars = pars[self.nTotalParCovariance:]
		
		self.covariance(covPars)

# Compute covariance matrix and invert it
		C = self.K + self.noiseLevel**2 * np.identity(self.nWavelengths)

		CInv, logD = cholesky.cholInvert(C)

# Call the model for the mean				
		w, S, dS = self.synth.synthDerivatives(funPars)
		
		S = S[0,:]
		dS = dS[0,:,:]
		
		residual = self.stokes[0,:] - S
		likelihood = 0.5 * np.dot(np.dot(residual.T,CInv),residual) + 0.5 * logD
		
		alpha = np.dot(CInv, residual)
		
# Computation of the Jacobian
# First for the parameters of the covariance function
		jacobian = np.zeros(self.nTotalPars)
		
		for i in range(self.nTotalParCovariance):
			jacobian[i] = -0.5 * np.sum(numpy.core.umath_tests.inner1d(CInv, self.dK[:,:,i].T)) + 0.5*np.dot(np.dot(alpha.T,self.dK[:,:,i]),alpha)

# Chain rule to take into account that we use the exponential of the parameters			
		jacobian[0:self.nTotalParCovariance] *= -covPars
				
# And then the Jacobian of the mean
		for i in range(self.nParsModel):
			jacobian[self.nTotalParCovariance+i] = -np.dot(alpha.T,dS[i,:])
			
		return likelihood, jacobian		