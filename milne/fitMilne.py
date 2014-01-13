import numpy as np
import matplotlib.pyplot as pl
import milne
import cholesky

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
		self.K = 0.0
		self.dK = []
		for i in range(self.nCovariance):
			right = left + self.nParCovariance[i]
			pars = covPars[left:right]
			K, dK = self.covTypes[self.funCovariance[i]](pars)
			left = right
			self.K += K
			self.dK.append(dK)
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
	
		return K, np.vstack((dKdlambdaGP, dKdsigmaGP))
	
	def marginalLikelihood(self, pars):
		"""
		"""
		
		covPars = np.exp(pars[0:self.nTotalParCovariance])
		funPars = pars[self.nTotalParCovariance:]
		
		self.covariance(covPars)
									
		C = self.K + self.noiseLevel**2 * np.identity(self.lineInfo[-1])

		CInv, logD = cholesky.cholInvert(C)
				
		w, S, dS = self.synth.synthDerivatives(funPars)
		
		S = S[0,:]
		dS = dS[0,:,:]
		
		residual = self.stokes[0,:] - S
		likelihood = 0.5 * np.dot(np.dot(residual.T,CInv),residual) + 0.5 * logD
		
	## Jacobian
		#jac = np.zeros(5)
		
	## dLdlambda
		#residual2 = np.dot(CInv, residual)
		#jac[0] = -0.5 * np.sum(numpy.core.umath_tests.inner1d(CInv, dKdl.T)) + 0.5*np.dot(np.dot(residual2.T,dKdl),residual2)
		#jac[0] = -jac[0] * lambdaGP
		
	## dLdsigma
		#jac[1] = -0.5 * np.sum(numpy.core.umath_tests.inner1d(CInv, dKds.T)) + 0.5*np.dot(np.dot(residual2.T,dKds),residual2)
		#jac[1] = -jac[1] * sigmaGP
		
		#for i in range(3):
			#jac[i+2] = -np.dot(residual2.T,jacVoigt[i,:])
				
		#return likelihood, jac

			

		