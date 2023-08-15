import numpy as np
from pymwalib.voltage_context import VoltageContext
import bifrost as bf
import glob
from astropy.io import fits
from pymwalib.metafits_context import MetafitsContext
#from your import Your, Writer
#from your.formats.filwriter import make_sigproc_object
from astropy.time import Time


directory = '/astro/mwavcs/asvo/668663/' # the directory of voltage files
metafits = '/astro/mwavcs/asvo/668663/1360706048.metafits' # metafits file
paths = sorted(glob.glob(str(directory) + '*_*_*.sub')) # paths to voltage files
Obs_ID = 1360706048

# find out all flagged tiles, including from the metafits file and the calibration solution
flag=[]
with MetafitsContext(metafits) as context:
    for r in context.rf_inputs:
        if r.flagged:
            flag.append(r.ant) # the MWAX voltage data is ordered by the "Antenna" field
flag=np.array(flag) # antennas from the metafits file
#flag_cal=np.array([39, 46, 47, 53, 136]) # antennas from the calibration solution
#flag=np.concatenate((flag,flag_cal))
## give the index of the sub-array
#sub_array=np.array([48,49,82,83,84,85,86,87,88,89,90,91,95,99,100,105,106,110,111, 112, 117, 118, 122, 123, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,139, 140, 141, 142, 143])
#full_array=np.arange(144)
#flag_arrays=[i for i in full_array if i not in sub_array]
#flag_arrays=np.array(flag_arrays)
#flag=np.concatenate((flag,flag_arrays))
flag=np.unique(flag) # combine all flagged tiles and remove duplicates


# read in voltage data
coarse_chan = 0 # for one coarse channel
context = VoltageContext(str(metafits), paths) # define the voltage context
dt = 896 # time span in seconds
frames = 1000 # the number of frames to split the 1sec of data, i.e. 1000 frames in 1sec correspond to time resolution of 1ms
detected_data=np.zeros((dt*frames, 1280000//frames)) # create an array in the shape of (no. of samples)*(no. of fine channels) in the coarse channel
for i in range(dt): # time span in seconds
    data = context.read_second(Obs_ID+i, 1, coarse_chan) # read in 1s of data, containing 20 50ms blocks
    data = data.reshape((20, 144, 2, 64000, 2)) # reshape the data in the structure of 50ms blocks*native samples*antennas*pol*(real+imaginary parts)
    data = data[:,:,:,:,0]+data[:,:,:,:,1]*1j
    data = data.astype('complex64')
    for k in range(20): # fft on each block
        data_block=data[k,:,:,:]
        data_block=data_block.reshape((144, 2, frames//20, 1280000//frames))
        data_gpu = bf.asarray(data_block, space='cuda')
        fft_of_data_gpu = bf.ndarray(shape=data_gpu.shape, dtype='cf32', space='cuda')
        f = bf.fft.Fft()
        f.init(data_gpu, fft_of_data_gpu, axes=3, apply_fftshift=True) # do fft along the 'frame' axis
        f.execute(data_gpu, fft_of_data_gpu)
        fft_of_data_cpu = fft_of_data_gpu.copy(space='system')
        detected_fft_data = np.abs(fft_of_data_cpu)**2 # calculate power
        for j in range(len(flag)):
            detected_fft_data[flag[j],:,:,:]=0 # make the power from all flagged tiles zero
        detected_fft_data = np.sum(detected_fft_data, axis=0) # sum up the power from 144 tiles
        detected_fft_data = np.sum(detected_fft_data, axis=0) # sum up the X and Y polarisations
        detected_data[frames*i+frames//20*k:frames*i+frames//20*(k+1),:]=detected_fft_data

# write detected_data to a fits file, and convert it to psrfits file on my desktop using your. Need to discuss with Danny how to write out psrfits file using bifrost
header = fits.open(str(metafits))[0].header
fits.writeto(str(directory) + str(Obs_ID) + '_ch' + str(132+coarse_chan) + '.fits', detected_data, header, overwrite=True)

##! write detected_data to a PSRFITS file using your. (discuss with Danny how to use bifrost to do this)
## make a header for the PSRFITS file
#t_obs = Time(Obs_ID, format='gps')
#MJD = t_obs.mjd # calculate the MJD time
#sigproc_object = make_sigproc_object(
#    rawdatafile  = str(Obs_ID) + '_ch' + str(109+coarse_chan) + '.fil',
#    source_name = str(Obs_ID),
#    nchans  = 1280,
#    foff = 0.001, #MHz
#    fch1 = (109+coarse_chan)*1.28-0.64, # MHz, center frequency of the coarse channel
#    tsamp = 0.001, # seconds
#    tstart = MJD, #MJD
#    src_raj=112233.44, # HHMMSS.SS
#    src_dej=112233.44, # DDMMSS.SS
#    machine_id=0,
#    nbeams=0,
#    ibeam=0,
#    nbits=32,
#    nifs=1,
#    barycentric=0,
#    pulsarcentric=0,
#    telescope_id=0,
#    data_type=1,
#    az_start=-1,
#    za_start=-1,
#)
#sigproc_object.write_header(str(directory) + str(Obs_ID) + '_ch' + str(109+coarse_chan) + '.fil')
#detected_data = np.float32(detected_data)
#sigproc_object.append_spectra(detected_data, str(directory) + str(Obs_ID) + '_ch' + str(109+coarse_chan) + '.fil')
##new_object = Your(str(directory) + str(Obs_ID) + '_ch' + str(109+coarse_chan) + '.fil')
##writer_object = Writer(
##    new_object,
##    nstart=0,
##    nsamp=dt*frames,
##    c_min=0,
##    c_max=1280,
##    outdir=str(directory),
##    outname=str(Obs_ID) + '_ch' + str(109+coarse_chan),
##)
##writer_object.to_fits()
