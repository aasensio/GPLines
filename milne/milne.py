import milneFast as m
import numpy as np

class milne:
	"""
	Class that synthesizes spectral lines using Milne Eddington.
	To use it:
	from milne import milne as milne
	line = milne(lineInfo)
	lineInfo = [lambda0, JUp, JLow, gUp, gLow, lambdaStart, lambdaStep, nLambda]
	wavelength, stokes = line.synth(model)
	model = [BField, theta, chi, vmac, damping, beta, mu, doppler, kl]
	"""
	
	def __init__(self, lineInfo):
		self.lineInfo = lineInfo		
		m.milnemod.setline(lineInfo)
		self.nPars = 9
		
	
	def synth(self, model):
		"""
		Synthesize a spectral line using the Milne-Eddington model
		"""
		wavelength, stokes = m.milnemod.milnesynth(model,self.lineInfo[-1])
		
		return wavelength, stokes
	
	def __perturbParameter(self, model, index, relativeChange):
		
		newModel = model[:]
		if (model[index] == 0):
			change = 1.0e-3			
		else:
			change = model[index] * relativeChange
		
		newModel[index] = model[index] + change
				
		return newModel, change
			
	def synthDerivatives(self, model, relativeChange=1e-3):
		"""
		Compute the derivative of the Stokes profiles with respect to all the variables
		"""
		
		wavelength, stokes = m.milnemod.milnesynth(model,self.lineInfo[-1])
		
		stokesDeriv = np.zeros((9,4,self.lineInfo[-1]))
		
		for i in range(9):			
			newModel, change = self.__perturbParameter(model, i, relativeChange)			
			wavelength, stokesNew = m.milnemod.milnesynth(newModel,self.lineInfo[-1])
		
			stokesDeriv[i,:,:] = (stokesNew - stokes) / change			
		
		return wavelength, stokes, stokesDeriv
		