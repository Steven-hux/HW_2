
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('train_data_10.fits')
num = 5001 # the 5001st spectra in this fits file

flux = hdulist[0].data[num-1]  # 索引从0开始，索引5000就是5001个
objid = hdulist[1].data['objid'][num-1]
label = hdulist[1].data['label'][num-1]
wavelength = np.linspace(3900,9000,3000) # 3900到9000插值3000个点

c = {0:'GALAXY',1:'QSO',2:'STAR'}
plt.plot(wavelength,flux)
plt.title(f'class:{c[label]}')
plt.xlabel('wavelength ({})'.format(f'$\AA$'))
plt.ylabel('flux')
plt.show()

# 统计各个元素数量
fluxs = hdulist[0].data[::]  # 索引从0开始，索引5000就是5001个
objids = hdulist[1].data['objid'][::]
labels = hdulist[1].data['label'][::]
unique_elements, counts_elements = np.unique(labels, return_counts=True)

# 输出结果
print("Unique elements:", unique_elements)
print("Counts of each element:", counts_elements)