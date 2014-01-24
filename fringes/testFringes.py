import fitFringes
import numpy as np
import matplotlib.pyplot as pl
import fringe

model = [0.0, 0.05, 0.5, 0.0]

nWavelength = 150
noiseLevel = 1e-3
wavelength = np.linspace(0.0, 100.0, nWavelength)
fringeComponent = model[0] + model[1] * np.cos(model[2] * wavelength + model[3])
lineComponent = - 0.7*np.exp(-(wavelength-30.0)**2/2.0**2) - 0.4*np.exp(-(wavelength-75.0)**2/2.0**2)
stokes = fringeComponent + lineComponent
stokes += noiseLevel * np.random.randn(nWavelength)
stokes = stokes - np.mean(stokes)

GP = fitFringes.fringeGP(wavelength, stokes, noiseLevel)
GP.addCovariance('sqr')

GP.optimizeGPDIRECT()
resDIRECT = GP.predictGP(GP.optimalPars)

pl.close('all')
nRows = 2
nCols = 2
fig = pl.figure(figsize=(12,8))
ax = fig.add_subplot(nRows, nCols, 1)
ax.plot(wavelength, stokes)
ax.plot(wavelength, resDIRECT[1])

ax = fig.add_subplot(nRows, nCols, 2)
ax.plot(wavelength, resDIRECT[1]-resDIRECT[0])
ax.plot(wavelength, lineComponent)

ax = fig.add_subplot(nRows, nCols, 3)
ax.plot(wavelength, resDIRECT[0])
ax.plot(wavelength, fringeComponent)
#DIRECTPars = GP.optimalPars[:]

#GP.optimizeGP()
#res = GP.predictGP(GP.optimalPars)


#nRows = 3
#nCols = 4

#pl.close('all')
#fig = pl.figure(figsize=(16,8))
#a = np.arange(nRows*nCols)+1
#order = np.hstack(a.reshape(nRows, nCols).T)

#loop = 0
#for indStokes in range(4):
	#ax = fig.add_subplot(nRows,nCols,order[loop])
	
	#ax.plot(GP.wavelength, stokesNew[indStokes,:],'o', color='#969696')
	#ax.plot(GP.wavelength, res[1][indStokes,:], linewidth=2, color='#507FED')
	#ax.plot(GP.wavelength, resDIRECT[1][indStokes,:], linewidth=2, color='r')

	#loop += 1
	
	#ax = fig.add_subplot(nRows,nCols,order[loop])
	#ax.plot(GP.wavelength, stokes[indStokes,:],'o', color='#969696')
	#ax.plot(GP.wavelength, res[0][indStokes,:], linewidth=2, color='#507FED')
	#ax.plot(GP.wavelength, resDIRECT[0][indStokes,:], linewidth=2, color='r')
	
	#loop += 1
	
	#ax = fig.add_subplot(nRows,nCols,order[loop])
	#ax.plot(GP.wavelength, res[1][indStokes,:]-res[0][indStokes,:],'o', color='#969696')
	#ax.plot(GP.wavelength, resDIRECT[1][indStokes,:]-resDIRECT[0][indStokes,:],'o', color='red')
	#ax.plot(GP.wavelength, stokesNew[indStokes,:]-stokes[indStokes,:], linewidth=2, color='#507FED')

	#loop += 1
	
#fig.tight_layout()

#for i in range(8):
	#print "{0:.2f} - {1:.2f} - {2:.2f}".format(model[i], GP.optimalPars[2+i], DIRECTPars[2+i])