#!/usr/bin/env python3

# Make diagnostic plots for all idealized stellar kinematic maps

import matplotlib
matplotlib.use('pdf')
import os
from glob import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from time import process_time 
from copy import copy
import multiprocessing as mp
import random
import pymysql
from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

def add_colorbar(im,ax,label,colour='white'):
    axin = ax.inset_axes(
            [0.05,0.05, 0.9, 0.025], transform=ax.transAxes)
    cbar = plt.colorbar(im, cax=axin, orientation='horizontal')
    cbar.ax.tick_params(color=colour, labelcolor=colour,direction='in',labeltop=1,
                        labelbottom=0,top=1,bottom=0, width=1, labelsize=14)
    cbar.set_label(label, color=colour, size=12)
    cbar.ax.xaxis.set_label_position('top') 
    cbar.outline.set_edgecolor(colour)
    cbar.outline.set_linewidth(1.5)

def generate_figure(outfile,subhaloID,camera,snapID,use_sql=False,cnf_path=None):
    
    data_dir = '/home/bottrell/scratch/Merger_Kinematics/Moments/Illustris-TNG/TNG100-1/{:03}'.format(snapID)
    
    fig,axarr = plt.subplots(1,3,figsize=(15,5))
    fig.subplots_adjust(wspace=0.05,hspace=0.05)
    axarr = axarr.flat

    for ax in axarr:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ['top','left','right','bottom']:
            ax.spines[spine].set_linewidth(4)

    __mapfile_name__ = '{}/moments_TNG100-1_{}_{}_{}_{}__32.fits'

    ptype = 'stars'
    mapfile_name = __mapfile_name__.format(data_dir,snapID,subhaloID,ptype,camera)
    maps = fits.getdata(mapfile_name)
    header = fits.getheader(mapfile_name)
    fov_kpc = header['FOV_KPC']
    npixels = header['NAXIS1']
    kpc_per_pixel = fov_kpc/npixels
    
    mask = np.log10(maps[0])
    #maps[:,mask<4.5] = np.nan
    
    if not use_sql:
        mstar = np.log10(np.nansum(maps[0]))
    else:
        db = pymysql.connect(host='lauca.phys.uvic.ca',user='cbottrell',
                             database='IllustrisTNG100_1',read_default_file=cnf_path)
        c = db.cursor()
        dbcmd = 'SELECT Mstar,Tpostmerger,rsep,MassRatio FROM Environment WHERE DB_ID="{}_{}"'.format(snapID,subhaloID)
        c.execute(dbcmd)
        data = np.array(c.fetchall())[0]
        c.close()
        db.close()
        mstar=data[0]
        tpost=data[1]
        rsep=data[2]
        mu=data[3]
        if rsep>300: rsep=999

    ax = axarr[0]
    cmap = copy(plt.cm.bone)
    cmap.set_bad('black', 1.0)
    vmin = np.around((4.5-np.log10(kpc_per_pixel**2))*2)/2
    vmax = np.around(np.nanpercentile(np.log10(maps[0]/kpc_per_pixel**2),99.9)*2)/2
    im = ax.imshow(np.log10(maps[0]/kpc_per_pixel**2),vmin=vmin,vmax=vmax,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar surface density, $\log\;\Sigma_{\star}$ [M$_{\odot}$/kpc$^2$]'
    add_colorbar(im,ax,label,colour=cbar_colour)
    
    txt = ['subhaloID: {}\n'.format(subhaloID),
           '$\log($M$_{{\star}}/$M$_{{\odot}}$): {:0.2f}\n'.format(mstar)]
    if use_sql:
        txt += ''.join(['$t_{{pm}}$: {:0.2f} Gyr\n'.format(tpost),
                        '$r_{{sep}}$: {:.1f}\n'.format(rsep),
                        '$\mu$: {:.2f}'.format(mu)])
    ax.text(0.025,0.975,''.join(txt),transform=ax.transAxes,ha='left',va='top',color='white',size=12)
    ax.axhline(y=0.975*maps.shape[1],xmax=0.975,xmin=0.725,c='white',lw=2)
    ax.text(0.975,0.96,'{:.1f} kpc'.format(2.5*fov_kpc/10),ha='right',va='top',color='white',size=12,transform=ax.transAxes)
    
    ax = axarr[1]
    cmap = copy(plt.cm.jet)
    cmap.set_bad('black', 1.0)
    vmap = copy(maps[1])
    vmap[mask<5]=np.nan
    vmin = np.abs(np.nanpercentile(vmap[192:-192,192:-192],1))
    vmax = np.abs(np.nanpercentile(vmap[192:-192,192:-192],99))
    vmax = (vmin+vmax)/2
    vmin = -vmax
    im = ax.imshow(maps[1],vmin=vmin,vmax=vmax,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar velocity, $V_{\mathrm{LOS,\star}}$ [km/s]'
    add_colorbar(im,ax,label,colour=cbar_colour)

    ax = axarr[2]
    cmap = copy(plt.cm.Blues_r)
    cmap.set_bad('black', 1.0)
    smap = copy(maps[2])
    smap[mask<5]=np.nan
    smin = np.abs(np.nanpercentile(smap[192:-192,192:-192],1))
    smax = np.abs(np.nanpercentile(smap[192:-192,192:-192],99))
    im = ax.imshow(maps[2],vmin=smin,vmax=smax,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar vel. dispersion, $\sigma_{\mathrm{LOS,\star}}$ [km/s]'
    add_colorbar(im,ax,label,colour=cbar_colour)
    fig.savefig(outfile,bbox_inches='tight')
    plt.close(fig)
    

def main():
    
    snapID = 99
    data_dir = '/home/bottrell/scratch/Merger_Kinematics/Moments/Illustris-TNG/TNG100-1/{:03}'.format(snapID)
    save_dir = '/home/bottrell/scratch/Merger_Kinematics/Figures/TNG100-1/Diagnostic_Stellar_Ideal'
    filenames = list(sorted(glob('{}/moment*.fits'.format(data_dir))))
    subhaloIDs = list(sorted(set(([filename.split('_')[-5] for filename in filenames]))))
    cameras = ['i0','i1','i2','i3']

    for subhaloID in subhaloIDs:
        for camera in cameras:
            save_dir = '/home/bottrell/scratch/Merger_Kinematics/Figures/TNG100-1/Diagnostic_Stellar_Ideal'
            outfile = '{}/TNG100-1_Diagnostic_Stellar-Ideal_snapNum-{}_subhaloID-{}_camera-{}.pdf'.format(save_dir,snapID,subhaloID,camera)
            if not os.access(outfile,0):
                generate_figure(outfile,subhaloID,camera,snapID,use_sql=True,cnf_path='/home/bottrell/.mysql/lauca.cnf')
            
    
if __name__ == '__main__':
    main()
