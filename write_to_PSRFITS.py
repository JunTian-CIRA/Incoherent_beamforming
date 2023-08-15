from your import Your, Writer
from your.formats.filwriter import make_sigproc_object
import numpy as np
from astropy.io import fits
from astropy.time import Time

Obs_ID = 1360706048
t_obs = Time(Obs_ID, format='gps')
MJD = t_obs.mjd # calculate the MJD time
directory = '/astro/mwavcs/asvo/668663/'
chan0 = 132
data_obs=np.zeros((896000, 128*24))
for i in range(24):
    hdu = fits.open(str(directory) + str(Obs_ID) + '_ch' + str(chan0+i) + '.fits')
    data = hdu[0].data
    hdu.close()
    data=np.average(data.reshape(896000, 128, 10), axis=2)
    data_obs[:, 128*i:128*(i+1)]=data
sigproc_object = make_sigproc_object(
rawdatafile  = str(Obs_ID) + '_ch' + str(chan0) + '.fil',
source_name = str(Obs_ID),
nchans  = 128*24,
foff = 0.01, #MHz
fch1 = 168.32, # MHz, center frequency of the 1st coarse channel
tsamp = 0.001, # seconds
tstart = MJD, #MJD
src_raj=112233.44, # HHMMSS.SS
src_dej=112233.44, # DDMMSS.SS
machine_id=0,
nbeams=0,
ibeam=0,
nbits=32,
nifs=1,
barycentric=0,
pulsarcentric=0,
telescope_id=0,
data_type=1,
az_start=-1,
za_start=-1,
)
data_written=data_obs
sigproc_object.write_header(str(directory) + str(Obs_ID) + '_ch' + str(chan0) + '.fil')
sigproc_object.append_spectra(np.float32(data_written), str(directory) + str(Obs_ID) + '_ch' + str(chan0) + '.fil')

new_object = Your(str(directory) + str(Obs_ID) + '_ch' + str(chan0) + '.fil')
writer_object = Writer(
new_object,
nstart=0,
nsamp=896000,
outdir=str(directory),
outname=str(Obs_ID) + '_ch_' + str(chan0),
)
writer_object.to_fits()
