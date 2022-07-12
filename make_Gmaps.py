from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt 
from scipy import interpolate
from matplotlib.widgets import Cursor
from matplotlib.gridspec import GridSpec

pad = '/Users/peeters/data/Analyse/RoyZhang/Summer2017/Data/'

### read 3dim cube (containing spectra of NGC2023 South)

hdulist = fits.open(pad+'NGC2023_SPECTRAL_MAP_SOUTH.fits')
# hdulist.info()
####		No.    Name      Ver    Type      Cards   Dimensions   Format
####		  0  PRIMARY       1 PrimaryHDU     297   (27, 17, 194)   float32   --> (x, y, lambda)
####		  1  WCS-TAB       1 BinTableHDU     13   1R x 1C   [194E]   

## read wavelength data and check type/shape
wave = hdulist[1].data['wavelength']
# print(type(wavelength))  #### <class 'numpy.ndarray'>
# print(wavelength.shape)  #### (1, 194, 1)

## put wavelength in 1dim array & check result
#           long version: wavelength = np.reshape(wavelength, len(wavelength[0,:,0]))
wave = np.reshape(wave, -1)
# print(type(wavelength))    #### <class 'numpy.ndarray'>
# print(wavelength.shape)    #### (194,)

## read data and check type/shape
# print(hdulist[0].header)
# print(hdulist[0].data.shape)   #### (194, 17, 27)
# print(type(hdulist[0].data))   #### <class 'numpy.ndarray'>
spectra = hdulist[0].data

## put in order of (x, y, lambda)
# print(np.swapaxes(spectra, 1, 2).shape) 	## swap dim 1 and 2 --> (194, 27, 17)
# print(np.moveaxis(np.swapaxes(spectra, 1, 2), 0, 2).shape) 	## put dim 0 to last place --> (27, 17, 194)
spectra = np.moveaxis(np.swapaxes(spectra, 1, 2), 0, 2) 	#### (27, 17, 194)

## close hdu list
hdulist.close()


### read 3dim continua

hdulist = fits.open(pad+'NGC2023_CONTINUUM_MAP_SOUTH.fits')
# hdulist.info()
####		No.    Name      Ver    Type      Cards   Dimensions   Format
####		  0  PRIMARY       1 PrimaryHDU     298   (27, 17, 194)   float64   --> (x, y, lambda)
####		  1  WAVELENGTH    1 BinTableHDU     11   89046R x 1C   [1E]   

wave_cont = hdulist[1].data['wavelength']
# print(type(wave_cont))	#### <class 'numpy.ndarray'>
# print(wave_cont.shape)    #### (89046,)
## wavelength array repeated for each pixel. Reshape to same format as the spectral cube 
##    -- order 'F' indicates the first index varies fastest, the last one slowest; order = 'C' is opposite, default = C
wave_cont = np.reshape(wave_cont, [27, 17, 194], order ='F')

## make integer array and test result!
# test = np.arange(194)
# print(wave_cont.shape)
# plt.plot(test, wave_cont[10,10,:], color='b')
# plt.show()
# plt.clf()

## read data and check type/shape
# print(hdulist[0].header)
# print(hdulist[0].data.shape)		#### (194, 17, 27)
# print(type(hdulist[0].data))		#### <class 'numpy.ndarray'>
cont = hdulist[0].data 		####(194, 17, 27)
cont = np.moveaxis(np.swapaxes(cont, 1, 2), 0, 2) 	#### (27, 17, 194)

## close hdu list
hdulist.close()


### read extinction map

hdulist = fits.open(pad+'NGC2023_EXTINCTION_MAPS_SOUTH.fits')
# hdulist.info()
#### 		No.    Name      Ver    Type      Cards   Dimensions   Format
####		  0  PRIMARY       1 PrimaryHDU     297   (27, 17)   float64   
#####		  1                1 ImageHDU         8   (27, 17, 194)   float64   
#####		  2  WAVELENGTH    1 BinTableHDU     11   194R x 1C   [1E]   

## read wavelength array
wave_ext = hdulist[2].data['wavelength']
#print(wave_ext.shape)    #### (194, )

## read data 
ext = hdulist[1].data
# print(ext.shape)		##### (194, 17, 27)
ext = np.moveaxis(np.swapaxes(ext, 1, 2), 0, 2) 	#### (27, 17, 194)
#### most pixels: extinction = 1 (data/ext == ext corrected spectra)

## check if all wavelength arrays are the same ---> Yes!
# plt.plot(wave, wave-wave_ext, color='r')
# plt.plot(wave, wave-wave_cont[15, 15, :], color='g')
# plt.plot(wave, wave_cont[4,4,:]-wave_cont[15, 15, :], color='b')
# plt.show()


## plot spectrum with corresponding continuum in pdf 
#### --> how to make a ps-file? how to make multiple figures on 1 page in a loop as this? (cf. !p.multi=[0,2,2])

# with PdfPages('plot_spectra_cont_C.pdf') as pdf:
# 	for i in np.arange(len(spectra[:, 0, 0])):
# 		for j in np.arange(len(spectra[0, :, 0])):
# 			plt.figure(figsize=(3, 3))
# 			plt.plot(wave, spectra[i, j, :], color='b')
# 			plt.plot(wave, cont[i, j, :], color='r')
# 			plt.plot(wave, ext[i,j,:]*100, color='g')
# 			plt.title('('+str(i)+','+str(j)+')')
# 			pdf.savefig()  # saves the current figure into a pdf page
# 			plt.close()
	

## correct spectra for extinction
extcorr_spectra = spectra/ext

## fit spline continuum 

## a) determine anchor points

## plot spectra with cursor (cursor doesn't work if you zoom in!)
# fig  = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, facecolor='#FFFFCC')
# ax.plot(wave, extcorr_spectra[18, 8, :], color='b')
# cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

flux = extcorr_spectra[18, 8, :]

anchor_ind_init = np.array([4, 10, 42, 57, 107, 115, 121, 145, 166, 178]) #### [5.36, 5.5, 6.54, 7.00, 9.39, 9.89, 10.26, 11.75, 13.06, 13.80]

## find closest wavelengths in wave array

xanchor = wave[anchor_ind_init]    #### <class 'numpy.ndarray'> (also array if you use a list for anchor_ind_init)
yanchor = flux[anchor_ind_init]

## b) determine spline fit

def splinefit(xpoints,ypoints):
  x = np.array(xpoints)
  y = np.array(ypoints)

  # spline interpolation
  tck = interpolate.splrep(x, y, s=0)

  # new wavelength points to evaluate spline
  xnew = np.arange(min(xpoints),max(xpoints),0.001)
  # evaluate spline at new points
  ynew = interpolate.splev(xnew, tck, der=0)

  # plot the spline
  # plt.plot(xnew,ynew,label='continuum')
  # plt.plot(xpoints,ypoints,'ro')
  # plt.show()

  # return
  return tck


spline = splinefit(xanchor, yanchor)
print(type(spline))
## c) test

plt.plot(wave, flux, color='b')
plt.plot(xanchor, yanchor, 'ro')
#plt.plot(xanchor, spline, 'g+')
plt.show()





i = 18
j = 8

print(((spectra-cont)/ext).shape)
t=(spectra-cont)
tt=(spectra-cont)/ext
plt.plot(wave, spectra[i, j, :], color='b')
plt.plot(wave, cont[i, j, :], color='r')
plt.plot(wave, ext[i,j,:]*100, color='g')
plt.plot(wave, t[i,j,:], color='y')
plt.plot(wave, tt[i,j,:], color='orange')
plt.show()

#


