import fitMilne
import numpy as np
import matplotlib.pyplot as pl
import milne

def genTestLine(lineInfo, model, noiseLevel=0.0):
	"""
	Generate an artificial spectral line that will be fitted later
	"""	
	synth = milne.milne(lineInfo)
	stokes = np.zeros((4,nLambda))
	wavelength, stokes = synth.synth(model)		
	stokes += noiseLevel * np.random.randn(4,nLambda)
	
	return wavelength, stokes


# Definition of the line
lambda0 = 6301.5080
JUp = 2.0
JLow = 2.0
gUp = 1.5
gLow = 1.833
lambdaStart = 6300.8
lambdaStep = 0.01
nLambda = 150

lineInfo = [lambda0, JUp, JLow, gUp, gLow, lambdaStart, lambdaStep, nLambda]

# Synthetic model
BField = 100.0
BTheta = 20.0
BChi = 20.0
VMac = 2.0
damping = 0.0
beta = 3.0
mu = 1.0
VDop = 0.15
kl = 5.0
model = [BField, BTheta, BChi, VMac, damping, beta, mu, VDop, kl]
	
noiseLevel = 1e-3
wavelength, stokes = genTestLine(lineInfo, model, noiseLevel)
wavelength -= np.mean(wavelength)

milne = fitMilne.milneGP(wavelength, stokes, noiseLevel, lineInfo)
milne.addCovariance('sqr')
milne.marginalLikelihood([1.0,2.0,BField, BTheta, BChi, VMac, damping, beta, mu, VDop, kl])


pl.plot(milne.wavelength, milne.stokes[3,:])