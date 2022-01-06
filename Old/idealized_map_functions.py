import os
import numpy as np
from astropy.io import fits

def generate_maps_losvdcube(losvd_data,vlim=700.):
    '''Compute moment maps for an LOSVD cube of shape (spatial_xels,spatial_yels,vels) where vels is the number of velocity elements. The velocity elements must be symmetric about zero with range [vlim,vlim] in km/s.'''
    vel = np.linspace(-vlim,vlim,losvd_data.shape[-1],endpoint=False)
    delv = 2*vlim/losvd_data.shape[-1]
    vel+=delv/2.
    sum_wi = np.nansum(losvd_data,axis=-1)
    sum_wivi = np.nansum(losvd_data*vel,axis=-1)
    vbar = sum_wivi/sum_wi
    Nprime = np.nansum(losvd_data>0,axis=-1)
    vstd = np.nansum(losvd_data*(vel-vbar[...,np.newaxis])**2,axis=-1)
    vstd /= (Nprime-1)/Nprime*sum_wi
    vstd = np.sqrt(vstd)
    losvd_maps = np.array([sum_wi,vbar,vstd])
    return losvd_maps 

def write_maps_to_fits(outname,losvd_maps,header):
    '''Write LOSVD moment maps to FITS file `outname` with primary header `header`.'''
    if os.access(outname,0): os.remove(outname)
    hdu_pri = fits.PrimaryHDU(losvd_maps)
    hdu_pri.header = header
    # !!! IF NOT ALREADY IN `header`, ADD ALL THE LOSVD HEADER KEYWORDS HERE !!! e.g.
    # header.append(('AUTHOR', 'MAAN HANI'),end=True)
    hdu_pri.writeto(outname)

def plot(maps):
    '''Plot maps.'''
    import matplotlib.pyplot as plt
    fig,axarr = plt.subplots(1,3,figsize=(15,5))
    ax = axarr[0]
    ax.imshow(np.log10(maps[0]),vmin=5,vmax=8,cmap='bone_r',origin='lower')
    ax = axarr[1]
    ax.imshow(maps[1],cmap='jet_r',vmin=-175,vmax=175,origin='lower')
    ax = axarr[2]
    ax.imshow(maps[2],cmap='Blues_r',vmin=50,vmax=150,origin='lower')
    return fig,axarr

def main():
    
    from shutil import copy as cp

    if 'SLURM_TMPDIR' in [key for key in os.environ.keys()]:
        wdir = os.environ['SLURM_TMPDIR']
        os.chdir(wdir)
        print(os.getcwd())
    
    simID = 'TNG100-1'
    snapID = 99
    subhaloID = 404216
    parttype = 'stars'
    camera = 0

    losvd_dir = '/project/6020225/bottrell/share/mhani/LOSVD'
    filename = 'losvd_{}_{}_{}_{}_i{}__32.fits'.format(simID,snapID,subhaloID,parttype,camera)
    if not os.access(filename,0):
        remote_filepath = '{}/{}'.format(losvd_dir,filename)
        cp(remote_filepath,wdir)

    datacube = fits.getdata(filename)
    hdr = fits.getheader(filename)
    vlim = hdr['VLIM']
    
    # generate maps from datacube (need vlim)
    maps = generate_maps_losvdcube(datacube,vlim=vlim)
    
    # name of output losvd moments fits file
    outname = '{}/moments_{}_{}_{}_{}_i{}__32.fits'.format(wdir,simID,snapID,subhaloID,parttype,camera)
    
    # write to fits file
    write_maps_to_fits(outname,maps,hdr)

    # plot results from file
    maps = fits.getdata(outname)
    fig,axarr = plot(maps)
    
if __name__ == '__main__':
    main()