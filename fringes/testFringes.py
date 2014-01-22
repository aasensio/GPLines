import fitMilne
import numpy as np
import matplotlib.pyplot as pl
import milne

model = [2.0, 0.5, 0.0]
	
noiseLevel = 1e-3
wavelength = np.linspace()
wavelength, stokes = 
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