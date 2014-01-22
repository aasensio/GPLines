import numpy as np

class fringe(object):
	
	def __init__(self, wavelength):
		self.wavelength = wavelength
		self.nWavelength = len(wavelength)
		pass
	
	def synthDerivatives(self, model):
		
		fringes = model[0] * np.sin(model[1] * wavelength + model[2])
		grad = np.zeros(3,self.nWavelength)
		grad[0] = np.sin(model[1] * wavelength + model[2])
		grad[1] = model[0] * wavelength * np.cos(model[1] * wavelength + model[2])
		grad[2] = model[0] * np.cos(model[1] * wavelength + model[2])
		
		return fringes, grad
		
		