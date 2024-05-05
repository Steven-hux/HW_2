# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:39:48 2024

@author: 99773
"""
import numpy as np
import matplotlib.pyplot as plt

def fitsread(filein):
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head


hdulist = fitsread('train_data_05.fits')
num = 5001 # the 5001st spectra in this fits file

flux = hdulist[0].data[num-1]
objid = hdulist[1].data['objid'][num-1]
label = hdulist[1].data['label'][num-1]
wavelength = np.linspace(3900,9000,3000)

c = {0:'GALAXY',1:'QSO',2:'STAR'}
plt.plot(wavelength,flux)
plt.title(f'class:{c[label]}')
plt.xlabel('wavelength ({})'.format(f'$\AA$'))
plt.ylabel('flux')
plt.show()