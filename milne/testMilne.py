import fitMilne
import numpy as np
import matplotlib.pyplot as pl
import milne

def genTestLine(lineInfo, model, mu=1.0, noiseLevel=0.0):
	"""
	Generate an artificial spectral line that will be fitted later
	"""	
	synth = milne.milne(lineInfo)
	stokes = np.zeros((4,nLambda))
	wavelength, stokes = synth.synth(model, mu)		
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
BField = 1000.0
BTheta = 20.0
BChi = 20.0
VMac = 2.0
damping = 0.0
beta = 3.0
VDop = 0.15
kl = 5.0
mu = 1.0
model = [BField, BTheta, BChi, VMac, damping, beta, VDop, kl]
	
noiseLevel = 1e-3
wavelength, stokes = genTestLine(lineInfo, model, mu, noiseLevel)
wavelength -= np.mean(wavelength)
stokesNew = stokes.copy()
stokesNew[0,:] += -0.5*wavelength


GP = fitMilne.milneGP(wavelength, stokesNew, noiseLevel, lineInfo, mu, whichToInvert=(True,True,True,True))
GP.addCovariance('sqr')

GP.optimizeGPDIRECT()
resDIRECT = GP.predictGP(GP.optimalPars)
DIRECTPars = GP.optimalPars[:]

GP.optimizeGP()
res = GP.predictGP(GP.optimalPars)


nRows = 3
nCols = 4

pl.close('all')
fig = pl.figure(figsize=(16,8))
a = np.arange(nRows*nCols)+1
order = np.hstack(a.reshape(nRows, nCols).T)

loop = 0
for indStokes in range(4):
	ax = fig.add_subplot(nRows,nCols,order[loop])
	
	ax.plot(GP.wavelength, stokesNew[indStokes,:],'o', color='#969696')
	ax.plot(GP.wavelength, res[1][indStokes,:], linewidth=2, color='#507FED')
	ax.plot(GP.wavelength, resDIRECT[1][indStokes,:], linewidth=2, color='r')

	loop += 1
	
	ax = fig.add_subplot(nRows,nCols,order[loop])
	ax.plot(GP.wavelength, stokes[indStokes,:],'o', color='#969696')
	ax.plot(GP.wavelength, res[0][indStokes,:], linewidth=2, color='#507FED')
	ax.plot(GP.wavelength, resDIRECT[0][indStokes,:], linewidth=2, color='r')
	
	loop += 1
	
	ax = fig.add_subplot(nRows,nCols,order[loop])
	ax.plot(GP.wavelength, res[1][indStokes,:]-res[0][indStokes,:],'o', color='#969696')
	ax.plot(GP.wavelength, resDIRECT[1][indStokes,:]-resDIRECT[0][indStokes,:],'o', color='red')
	ax.plot(GP.wavelength, stokesNew[indStokes,:]-stokes[indStokes,:], linewidth=2, color='#507FED')

	loop += 1
	
fig.tight_layout()

for i in range(8):
	print "{0:.2f} - {1:.2f} - {2:.2f}".format(model[i], GP.optimalPars[2+i], DIRECTPars[2+i])