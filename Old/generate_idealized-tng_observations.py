import os,sys,time,random
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from shutil import copy as cp
from copy import copy
from glob import glob
import RealSim_IFS
import warnings
from time import process_time 
import multiprocessing as mp
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import convolve
import pymysql 
import gc
import matplotlib
matplotlib.use('pdf')
rc('font',**{'family':'serif'})
rc('text', usetex=True)

def add_colorbar(im,ax,label,colour='white'):
    axin = ax.inset_axes(
            [0.05,0.05, 0.9, 0.025], transform=ax.transAxes)
    cbar = plt.colorbar(im, cax=axin, orientation='horizontal')
    cbar.ax.tick_params(color=colour, labelcolor=colour,direction='in',labeltop=1,
                        labelbottom=0,top=1,bottom=0, width=1, labelsize=16)
    cbar.set_label(label, color=colour, size=15)
    cbar.ax.xaxis.set_label_position('top') 
    cbar.outline.set_edgecolor(colour)
    cbar.outline.set_linewidth(1.5)

def plot(maps,header,vmins=None,vmaxs=None,return_vmaxima=False,show_cmap=True):
    
    fov_kpc = header['FOV_KPC']
    npixels = header['NPIXEL']
    subhaloID = header['SUBID']
    camera = header['AXIS']
    ptype = header['PARTTYPE']
    snapID = header['SNAPNUM']
    kpc_per_pixel = fov_kpc/npixels

    fig,axarr = plt.subplots(1,3,figsize=(15,5))
    fig.subplots_adjust(wspace=0.05,hspace=0.05)
    axarr = axarr.flat

    for ax in axarr:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ['top','left','right','bottom']:
            ax.spines[spine].set_linewidth(4)
    
    # Stellar Surface Density
    ax = axarr[0]
    cmap = copy(plt.cm.bone)
    cmap.set_bad('black', 1.0)
    if vmaxs is not None:
        m0min = vmins[0]
        m0max = vmaxs[0]
    else:
        m0min = np.around((4.5-2*np.log10(kpc_per_pixel))*2)/2
        m0max = np.around(np.nanpercentile(np.log10(maps[0]/kpc_per_pixel**2),99.9)*2)/2
    im = ax.imshow(np.log10(maps[0]/kpc_per_pixel**2),vmin=m0min,vmax=m0max,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar surface density, $\log\;\Sigma_{\star}$ [M$_{\odot}$/kpc$^2$]'
    if show_cmap:
        add_colorbar(im,ax,label,colour=cbar_colour)
  
    # Velocity
    ax = axarr[1]
    cmap = copy(plt.cm.jet)
    cmap.set_bad('black', 1.0)
    if vmaxs is not None:
        m1min = vmins[1]
        m1max = vmaxs[1]
    else:
        m1min = np.abs(np.nanpercentile(maps[1],1))
        m1max = np.abs(np.nanpercentile(maps[1],99))
        m1max = (m1min+m1max)/2
        m1min = -m1max
    im = ax.imshow(maps[1],vmin=m1min,vmax=m1max,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar velocity, $V_{\mathrm{LOS,\star}}$ [km/s]'
    if show_cmap:
        add_colorbar(im,ax,label,colour=cbar_colour)

    # Velocity dispersion
    ax = axarr[2]
    cmap = copy(plt.cm.Blues_r)
    cmap.set_bad('black', 1.0)
    if vmaxs is not None:
        m2min = vmins[2]
        m2max = vmaxs[2]
    else:
        m2min = np.abs(np.nanpercentile(maps[2],1))
        m2max = np.abs(np.nanpercentile(maps[2],99))
    im = ax.imshow(maps[2],vmin=m2min,vmax=m2max,cmap=cmap,origin='lower',aspect='auto',interpolation=None)
    cbar_colour = 'white'
    label = r'Stellar vel. dispersion, $\sigma_{\mathrm{LOS,\star}}$ [km/s]'
    if show_cmap:
        add_colorbar(im,ax,label,colour=cbar_colour)
    
    vmins = [m0min,m1min,m2min]
    vmaxs = [m0max,m1max,m2max]
    
    if return_vmaxima:
        return fig,axarr,vmins,vmaxs
    else:
        return fig,axarr
    
def add_ideal_info(ax,header):
    
    fov_kpc = header['FOV_KPC']
    npixels = header['NPIXEL']
    subhaloID = header['SUBID']
    camera = header['AXIS']
    ptype = header['PARTTYPE']
    snapID = header['SNAPNUM']
    kpc_per_pixel = fov_kpc/npixels
    
    # get ancillary data from database
    cnf_path='/home/bottrell/.mysql/lauca.cnf'
    db = pymysql.connect(host='lauca.phys.uvic.ca',
                         user='cbottrell',
                         database='IllustrisTNG100_1',
                         read_default_file=cnf_path)
    c = db.cursor()
    dbcmd = ['SELECT Mstar,Tpostmerger,rsep,MassRatio',
             'FROM Environment',
             'WHERE DB_ID="{}_{}"'.format(snapID,subhaloID)]
    c.execute(' '.join(dbcmd))
    data = np.array(c.fetchall())[0]
    c.close()
    db.close()
    mstar=data[0]
    tpost=data[1]
    rsep=data[2]
    mu=data[3]
    if rsep>300: rsep=999
        
    txts = [r'subhaloID: {} \\'.format(subhaloID),
            r'$\log($M$_{{\star}}/$M$_{{\odot}}$): {:0.2f} \\'.format(mstar),
            r'$t_{{pm}}$: {:0.2f} Gyr \\'.format(tpost),
            r'$r_{{sep}}$: {:.1f} \\'.format(rsep),
            r'$\mu$: {:.2f}'.format(mu)]
    offset = 0.0
    for txt in txts:
        ax.text(0.025,0.975-offset,''.join(txt),transform=ax.transAxes,ha='left',va='top',color='white',size=15)
        offset+=0.07
    ax.axhline(y=0.975*npixels,xmax=0.975,xmin=0.725,c='white',lw=2)
    ax.text(0.975,0.96,'{:.1f} kpc'.format(2.5*fov_kpc/10),ha='right',va='top',color='white',size=15,transform=ax.transAxes)

def overlay_ifu_design(fig,axarr,datacube,arcsec_per_pixel,manga_ifu_design='N127'):
    (xc_arcsec,yc_arcsec),params = RealSim_IFS.MaNGA_Observe(bundle_name=manga_ifu_design,
                                                   n_observations='Classic',
                                                   rotation_degrees = 0.,
                                                   return_params=True)
    core_diameter_arcsec = params['core_diameter_arcsec']
    xc_pixels = xc_arcsec/arcsec_per_pixel+datacube.shape[2]/2.
    yc_pixels = yc_arcsec/arcsec_per_pixel+datacube.shape[1]/2.
    core_diameter_pixels = core_diameter_arcsec/arcsec_per_pixel
    for ax in axarr:
        for xy in zip(xc_pixels.flat,yc_pixels.flat):
            core = Circle(xy=xy,radius=core_diameter_pixels/2,transform=ax.transData,edgecolor='Black',facecolor='None',lw=0.5)
            ax.add_artist(core)
    return fig,axarr

def apply_ifu_design(datacube,arcsec_per_pixel,manga_ifu_design):
    (xc_arcsec,yc_arcsec),params = RealSim_IFS.MaNGA_Observe(bundle_name=manga_ifu_design,
                                                             n_observations='Classic',
                                                             rotation_degrees = 0.,
                                                             return_params=True)
    core_diameter_arcsec = params['core_diameter_arcsec']
    xc_pixels = (xc_arcsec/arcsec_per_pixel+datacube.shape[2]/2.).flatten() # convert to pixels and center
    yc_pixels = (yc_arcsec/arcsec_per_pixel+datacube.shape[1]/2.).flatten() # convert to pixels and center
    core_diameter_pixels = core_diameter_arcsec/arcsec_per_pixel # convert to pixels
    
    core_arrays,weights = RealSim_IFS.Fiber_Observe(datacube,xc_pixels,yc_pixels,
                                                    core_diameter_pixels,return_weights=True)

    output_grid_dims = (100,100)
    out_arcsec_per_pixel = 0.5
    xc_pixels = xc_arcsec/out_arcsec_per_pixel + output_grid_dims[1]/2.
    yc_pixels = yc_arcsec/out_arcsec_per_pixel + output_grid_dims[0]/2.
    core_diameter_pixels = core_diameter_arcsec/out_arcsec_per_pixel
    
    outcube,weight_map = RealSim_IFS.Fiber_to_Grid(core_arrays,xc_pixels,yc_pixels,core_diameter_pixels,
                                               grid_dimensions_pixels=output_grid_dims,use_gaussian_weights=True,
                                               gaussian_sigma_pixels=0.7/out_arcsec_per_pixel,
                                               rlim_pixels=1.6/out_arcsec_per_pixel)
    outcube[outcube==0.]=np.nan
    return outcube,out_arcsec_per_pixel

def add_conv_info(ax,header_ideal,header_real):
    ax.axhline(y=0.975*header_ideal['NPIXEL'],xmax=0.975,xmin=0.725,c='white',lw=2)
    ax.text(0.975,0.96,'{:.1f} arcsec'.format(2.5*header_ideal['FOV_KPC']/10/header_real['SCALE_3']),ha='right',va='top',
            color='white',size=15,transform=ax.transAxes)
    txts = [r'Sample: {}'.format(header_real['SAMPLE']),
            r'Seeing: {:.2f} arcsec \\'.format(header_real['SEEING']),
            r'$z_{{\mathrm{{obs}}}}$: {:.4f} \\'.format(header_real['REDSHIFT']),
            r'IFU: {}'.format(header_real['DESIGN']),]
    offset = 0.0
    for txt in txts:
        ax.text(0.025,0.975-offset,''.join(txt),transform=ax.transAxes,ha='left',va='top',color='white',size=15)
        offset+=0.07
        
def add_real_info(ax,header_real):
    txts = [r'IFU Design: {}'.format(header_real['DESIGN']),
            r'R$_{{\mathrm{{IFU}}}}/$R$_{{\mathrm{{eff}}}}$: {:.2f}'.format(header_real['F_REFF']),]
    offset = 0.0
    for txt in txts:
        ax.text(0.025,0.975-offset,''.join(txt),transform=ax.transAxes,ha='left',va='top',color='white',size=15)
        offset+=0.07
    ax.axhline(y=0.975*header_real['NPIXEL'],xmax=0.975,xmin=0.725,c='white',lw=2)
    ax.text(0.975,0.96,'{:.1f} arcsec'.format(2.5*header_real['FOVSIZE']/10),ha='right',va='top',
            color='white',size=15,transform=ax.transAxes)

def main(subhaloID,snapID,ptype,camera,sample,seed=0):
    
    if 'SLURM_TMPDIR' in [key for key in os.environ.keys()]:
        wdir = os.environ['SLURM_TMPDIR']
        os.chdir(wdir)
    else: 
        wdir = os.getcwd()
    
#     output_dir = '/home/bottrell/scratch/Merger_Kinematics/RealSim-IFS/LOSVD/TNG100-1'
#     outcube_name = 'losvd_TNG100-1_{}_{}_{}_{}__32_MANGA_{}.fits'.format(snapID,subhaloID,ptype,camera,sample)
#     if os.access('{}/{}'.format(output_dir,outcube_name),0):
#         print('{}\nalready exists in output directory.'.format(outcube_name))
#         return None
    
    incube_name = 'losvd_TNG100-1_{}_{}_{}_{}__32.fits'.format(snapID,subhaloID,ptype,camera)
    if not os.access('{}/{}'.format(wdir,incube_name),0):
        losvd_dir = '/home/bottrell/projects/def-simardl/bottrell/share/mhani/LOSVD'
        cp('{}/{}'.format(losvd_dir,incube_name),wdir)
        
    figure_dir = '/home/bottrell/scratch/Merger_Kinematics/RealSim-IFS/Figures/TNG100-1'
    
#     print('\n')
#     print('-'*50)
#     print('\n')
#     print('SNAPNUM: {}'.format(snapID))
#     print('SUBHALOID: {}'.format(subhaloID))
#     print('CAMERA: {}'.format(camera))
#     print('PTYPE: {}'.format(ptype))
#     print('SAMPLE: {}'.format(sample))
#     print('\n')
    
#     # Cosmology 
    
#     cosmo=FlatLambdaCDM(H0=67.4,Om0=0.315)
#     # physical spatial pixel scale in datacube
#     kpc_per_pixel = hdr['FOV_KPC']/hdr['NPIXEL']
#     print("FOV [kpc]:",hdr['FOV_KPC'])
#     # The FOVSIZE keyword gives the size of the FOV in
#     # units of FOVUNIT (in this case, stellar half-mass radii)
#     R_eff_kpc = hdr['FOV_KPC']/hdr['FOVSIZE']
    
#     # Redshifts from real galaxy distribution
    
#     zcat_dir = '/home/bottrell/scratch/Merger_Kinematics/RealSim-IFS/Resources'
    
#     if sample == 'PRI':
#         zcat_name = '{}/MaNGA_target_redshifts-{}.txt'.format(zcat_dir,'primary')
#     elif sample == 'SEC':
#         zcat_name = '{}/MaNGA_target_redshifts-{}.txt'.format(zcat_dir,'secondary')
#     elif sample == 'CEN':
#         zcat_name = '{}/MaNGA_target_redshifts-{}.txt'.format(zcat_dir,'cen')
#     else:
#         zcat_name = '{}/MaNGA_target_redshifts-{}.txt'.format(zcat_dir,'all')
#     redshift_pool = np.loadtxt(zcat_name)
    
#     # IFU design sizes from which to choose
    
#     N_Reff = (2.5 if sample=='SEC' else 1.5)
#     manga_ifu_designs = np.array(['N19','N37','N61','N91','N127'])
#     manga_ifu_diameters_arcsec = np.array([12.5,17.5,22.5,27.5,32.5]) # [arcsec]
    
#     redshift = RealSim_IFS.get_random_redshift_manga(redshift_pool=redshift_pool,seed=seed)
#     kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60. # [kpc/arcsec]
#     R_eff_arcsec = R_eff_kpc / kpc_per_arcsec # [arcsec]
#     ifu_design_arg = np.argmin(np.abs(2*N_Reff*R_eff_arcsec-manga_ifu_diameters_arcsec))
#     ifu_design = manga_ifu_designs[ifu_design_arg]
#     ifu_diameter_arcsec = manga_ifu_diameters_arcsec[ifu_design_arg]
    
#     # If the design does not cover out to at least 0.75*N_Reff, choose another redshift and return new seed
    
#     # Also check that the IFU size is not large than the full field of view
    
#     seeds = range(int(1e9))
#     while (((ifu_diameter_arcsec/2)/(R_eff_arcsec*N_Reff))<0.75) or (ifu_diameter_arcsec>=R_eff_arcsec*10):
#         seed = random.choice(seeds)
#         redshift = RealSim_IFS.get_random_redshift_manga(redshift_pool=redshift_pool,seed=seed)
#         kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60. # [kpc/arcsec]
#         R_eff_arcsec = R_eff_kpc / kpc_per_arcsec # [arcsec]
#         ifu_design_arg = np.argmin(np.abs(2*N_Reff*R_eff_arcsec-manga_ifu_diameters_arcsec))
#         ifu_design = manga_ifu_designs[ifu_design_arg]
#         ifu_diameter_arcsec = manga_ifu_diameters_arcsec[ifu_design_arg]
#     print("Redshift:",redshift)
    
#     # Draw atmospheric seeing from real galaxy distribution
    
#     seeing_fwhm_arcsec = RealSim_IFS.get_random_seeing_manga(seed=seed)
#     print("Seeing FWHM [arcsec]:",seeing_fwhm_arcsec)
#     arcsec_per_pixel = kpc_per_pixel/kpc_per_arcsec
#     print('FOV [arcsec]:', hdr['FOV_KPC']/kpc_per_arcsec)
#     R_eff_arcsec = R_eff_kpc / kpc_per_arcsec # [arcsec]
#     print('R_eff [arcsec]:', R_eff_arcsec)
#     print('IFU design:',ifu_design,'\n')
    
#     # Convolve datacube

#     print('Starting convolution...\n')
#     start = time.time()
#     datacube = RealSim_IFS.apply_seeing(datacube=datacube,kpc_per_pixel=kpc_per_pixel,redshift=redshift,
#                                        seeing_fwhm_arcsec=seeing_fwhm_arcsec,seeing_model='manga',
#                                        use_threading=False,n_threads=1,cosmo=cosmo)
#     print('Convolution time [s]: {}\n'.format(time.time()-start))
    
#     # generate ifu observation with design
    
#     print('Starting observation...\n')
#     if os.access(outcube_name,0):
#         os.remove(outcube_name) 
#     start = time.time()
#     outcube,out_arcsec_per_pixel = apply_ifu_design(datacube,arcsec_per_pixel,ifu_design)
#     print('Compute time for IFU "observation" [s]: {}\n'.format(time.time()-start))

#     # save to file
    
#     hdu_pri = fits.PrimaryHDU(outcube)
#     warnings.simplefilter('ignore', category=AstropyWarning)
#     hdr['FOVSIZE'] = (outcube.shape[1]*out_arcsec_per_pixel,'FOV size')
#     hdr['FOVUNIT'] = ('arcsec','FOV size units')
#     hdr['FOV_KPC'] = (outcube.shape[1]*out_arcsec_per_pixel*kpc_per_arcsec,'[kpc]')
#     hdr['NPIXEL'] = (outcube.shape[1],'spatial grid resolution')
#     hdr.append(('SAMPLE',sample,'MaNGA sample'),end=True)
#     hdr.append(('SEED',seed,'RealSim-IFS random seed'),end=True)
#     hdr.append(('REDSHIFT',redshift,'Redshift'),end=True)
#     hdr.append(('SEEING',seeing_fwhm_arcsec,'Seeing FWHM [arcsec]'),end=True)
#     hdr.append(('SEEMDL','MaNGA','Seeing Model'),end=True)
#     hdr.append(('FOV_ARC',outcube.shape[1]*out_arcsec_per_pixel,'Field of view [arcsec]'),end=True)
#     hdr.append(('DESIGN',ifu_design,'MaNGA IFU design'),end=True)
#     hdr.append(('IFUDIAM',ifu_diameter_arcsec,'IFU diameter [arcsec]'),end=True)
#     hdr.append(('F_REFF',ifu_diameter_arcsec/2/R_eff_arcsec,'R_IFU / Reff'),end=True)
#     hdr.append(('N_REFF',N_Reff,'Target N_Reff of Sample'),end=True)
#     hdr.append(('N_OBS','Classic','MaNGA dither pattern'),end=True)
#     hdr.append(('ROTATE',0.,'IFU rotation [degrees]'),end=True)
#     hdr.append(('COSMO','FLAT_LCDM','Cosmology'),end=True)
#     hdr.append(('OMEGA_M',cosmo.Om(0),'Matter density'),end=True)
#     hdr.append(('OMEGA_L',cosmo.Ode(0),'Dark energy density'),end=True)
#     hdr.append(('SCALE_1',out_arcsec_per_pixel,'[arcsec/pixel]'),end=True)
#     hdr.append(('SCALE_2',kpc_per_arcsec*out_arcsec_per_pixel,'[kpc/pixel]'),end=True)
#     hdr.append(('SCALE_3',kpc_per_arcsec,'[kpc/arcsec]'),end=True)
#     hdr.append(('LUMDIST',cosmo.luminosity_distance(z=redshift).value,'Luminosity Distance [Mpc]'),end=True)

#     hdu_pri.header = hdr
#     hdu_pri.writeto(outcube_name)
#     cp(outcube_name,output_dir)
    
    # Figure creation 
    
    _figname_ = '{}/losvd_TNG100-1_{}_{}_{}_{}__32_MANGA_{}.pdf' # .format(snapID,subhaloID,ptype,camera,sample)
    
    filename_ideal = copy(incube_name)
    data_ideal = fits.getdata(filename_ideal)
    header_ideal = fits.getheader(filename_ideal)
    maps_ideal = RealSim_IFS.Generate_Maps_From_File(filename_ideal)
    
#     filename_real = copy(outcube_name)
#     data_real = fits.getdata(filename_real)
#     header_real = fits.getheader(filename_real)
#     maps_real = RealSim_IFS.Generate_Maps_From_Data(filename_real,data_real)
    
    fig1,axarr1,vmins,vmaxs = plot(maps_ideal,header_ideal,return_vmaxima=True)
    add_ideal_info(axarr1[0],header_ideal)
    
    figname = _figname_.format(figure_dir,snapID,subhaloID,ptype,camera,'IDEAL')
    fig1.savefig(figname,bbox_inches='tight')
    
#     maps_conv = RealSim_IFS.apply_seeing(maps_ideal, header_ideal['FOV_KPC']/header_ideal['NPIXEL'],
#                                          redshift = header_real['REDSHIFT'], seeing_model='manga', 
#                                          seeing_fwhm_arcsec=header_real['SEEING'], 
#                                          cosmo=FlatLambdaCDM(H0=67.4,Om0=0.315), 
#                                          use_threading=False, n_threads=1)
    
#     fig2,axarr2 = plot(maps_conv,header_ideal,vmaxs=vmaxs,vmins=vmins)
#     add_conv_info(axarr2[0],header_ideal,header_real)
#     arcsec_per_pixel = header_ideal['FOV_KPC']/header_ideal['NPIXEL']/header_real['SCALE_3']
#     overlay_ifu_design(fig2,axarr2,maps_conv,arcsec_per_pixel,manga_ifu_design=header_real['DESIGN'])
    
#     figname = _figname_.format(figure_dir,snapID,subhaloID,ptype,camera,'CONV-{}'.format(header_real['SAMPLE']))
#     fig2.savefig(figname,bbox_inches='tight')
    
#     fig3,axarr3 = plot(maps_real,header_real,vmins=vmins,vmaxs=vmaxs)
#     add_real_info(axarr3[0],header_real)
    
#     figname = _figname_.format(figure_dir,snapID,subhaloID,ptype,camera,'REAL-{}'.format(header_real['SAMPLE']))
#     fig3.savefig(figname,bbox_inches='tight')
    
#     if os.access(incube_name,0): os.remove(incube_name)
#     if os.access(outcube_name,0): os.remove(outcube_name)
    
    
if __name__ == '__main__':
    
    gc.collect()
    
    # get full set of possible argument variables from filenames
    
    fileList = list(sorted(glob('/home/bottrell/scratch/Merger_Kinematics/LOSVD/TNG100-1/099/*stars*.fits')))
    snapIDs = [int(fileName.split('_')[-4]) for fileName in fileList]
    subhaloIDs = [int(fileName.split('_')[-3]) for fileName in fileList]
    ptypes = [fileName.split('_')[-2] for fileName in fileList]
    cameras = [fileName.split('_')[-1].split('.fits')[0] for fileName in fileList]
    
    # primary, secondary, and colour-enhanced
    samples = ['ALL',]
    
    argList = []
    for i in range(len(snapIDs)):
        for j in range(len(samples)):
            argList.append((snapIDs[i],subhaloIDs[i],ptypes[i],cameras[i],samples[j]))
            
    arg_index = int(sys.argv[1])
    
    print(len(argList))
    
    snapID = argList[arg_index][0]
    subhaloID = argList[arg_index][1]
    ptype = argList[arg_index][2]
    camera = argList[arg_index][3]
    sample = argList[arg_index][4]
    
    if sample not in samples:
        raise Exception('SAMPLE argument variable not in [PRI,SEC,CEN,ALL]. Stopping...')
        
    if ptype != 'stars':
        raise Exception('PTYPE is not `stars`. Stopping...')
    
    random.seed()
    seeds = range(int(1e9))
    seed = random.choice(seeds)
    
    main(subhaloID,snapID,ptype,camera,sample=sample,seed=seed)
