import numpy as np

class fringe(object):
	
	def __init__(self, wavelength):
		self.wavelength = wavelength
		self.nWavelength = len(wavelength)
		pass
	
	def synthDerivatives(self, model):
		
		fringes = model[0] + model[1] * np.sin(model[2] * self.wavelength + model[3])
		grad = np.zeros((4,self.nWavelength))
		grad[0,:] = 1.0
		grad[1,:] = np.sin(model[2] * self.wavelength + model[3])
		grad[2,:] = model[1] * self.wavelength * np.cos(model[2] * self.wavelength + model[3])
		grad[3,:] = model[1] * np.cos(model[2] * self.wavelength + model[3])
		
		return fringes, grad
	
	def synth(self, model):
		
		fringes = model[0] + model[1] * np.sin(model[2] * self.wavelength + model[3])
				
		return fringes
		
		