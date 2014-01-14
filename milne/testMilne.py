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
modelBlend = [0.0, 0.0, 0.0, VMac+3, damping, 0.5, mu, VDop, 1.0]
	
noiseLevel = 1e-2
wavelength, stokes = genTestLine(lineInfo, model, noiseLevel)
wavelength, stokesBlend = genTestLine(lineInfo, modelBlend, noiseLevel)
wavelength -= np.mean(wavelength)
stokesNew = stokes.copy()
stokesNew[0,:] += 0.5*wavelength


GP = fitMilne.milneGP(wavelength, stokesNew, noiseLevel, lineInfo)
GP.addCovariance('sqr')
GP.optimizeGP()
#GP.marginalLikelihood([2.0,1.0,BField, BTheta, BChi, VMac, damping, beta, mu, VDop, kl])
res = GP.predictGP([1.0,1.0,BField, BTheta, BChi, VMac, damping, beta, mu, VDop, kl])


fig = pl.figure(figsize=(12,8))
ax = fig.add_subplot(221)
ax.plot(GP.wavelength, stokesNew[0,:],'o', color='#969696')
ax.plot(GP.wavelength, res[1], linewidth=2, color='#507FED')

ax = fig.add_subplot(222)
ax.plot(GP.wavelength, stokes[0,:],'o', color='#969696')
ax.plot(GP.wavelength, res[0], linewidth=2, color='#507FED')

ax = fig.add_subplot(223)
ax.plot(GP.wavelength, res[1]-res[0],'o', color='#969696')
ax.plot(GP.wavelength, stokesNew[0,:]-stokes[0,:], linewidth=2, color='#507FED')

ax = fig.add_subplot(224)
ax.plot(GP.wavelength, stokesNew[0,:],'bo')
pl.plot(GP.wavelength, res[0], 'y', linewidth=2)
pl.plot(GP.wavelength, res[1], 'k', linewidth=2)
pl.plot(GP.wavelength, stokes[0,:], 'ro')