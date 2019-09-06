# Import packages
from __future__ import division, print_function
import os
import copy
import sep

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

from .imutils import extract_obj, make_binary_mask

from kungpao import imtools
from kungpao import io
from kungpao.display import display_single, IMG_CMAP, SEG_CMAP
from kungpao.galsbp import galSBP


# Define pixel scale of different surveys, unit = arcsec / pixel
HSC_pixel_scale = 0.168
DECaLS_pixel_scale = 0.262
Dragonfly_pixel_scale = 2.5
SDSS_pixel_scale = 0.395

# Define zeropoint of different surveys
HSC_zeropoint = 27.0
DECaLS_zeropoint = 22.5
SDSS_zeropoint = 22.5
Dragonfly_zeropoint_g = 27.37635915911822 
Dragonfly_zeropoint_r = 27.10646046539894

HSC_binray_mask_dict = {0: 'BAD',
                        1:  'SAT (saturated)',
                        2:  'INTRP (interpolated)',
                        3:  'CR (cosmic ray)',
                        4:  'EDGE (edge of the CCD)',
                        5:  'DETECTED',
                        6:  'DETECTED_NEGATIVE',
                        7:  'SUSPECT (suspicious pixel)',
                        8:  'NO_DATA',
                        9:  'BRIGHT_OBJECT (bright star mask, not available in S18A yet)',
                        10: 'CROSSTALK', 
                        11: 'NOT_DEBLENDED (For objects that are too big to run deblender)',
                        12: 'UNMASKEDNAN',
                        13: 'REJECTED',
                        14: 'CLIPPED',
                        15: 'SENSOR_EDGE',
                        16: 'INEXACT_PSF'}

SkyObj_aperture_dic = { '20': 5.0,
                        '30': 9.0,
                        '40': 12.0,
                        '57': 17.0,
                        '84': 25.0,
                        '118': 35.0 }

__all__ = ["skyobj_value", "evaluate_sky", "evaluate_sky_dragonfly", "run_SBP"]

#########################################################################
########################## 1-D profile related ##########################
#########################################################################

# Calculate mean/median value of nearby sky objects
def skyobj_value(sky_cat, cen_ra, cen_dec, matching_radius=[1, 3], aperture='84', 
    print_number=False, sigma_upper=3., sigma_lower=3., maxiters=5, showmedian=False):
    '''Calculate the mean/median value of nearby SKY OBJECTS around a given RA and DEC.
    Importing sky objects catalog can be really slow.

    Parameters:
    -----------
    path: string, the path of catalog.
    cen_ra, cen_dec: float, RA and DEC of the given object.
    matching_radius: float or list, in arcmin. We match sky objects around the given object within this radius/range.
    aperture: string, must be in the `SkyObj_aperture_dic`.
    print_number: boolean. If true, it will print the number of nearby sky objects.
    sigma_upper, sigma_lower: float, threshold for sigma_clipping of nearby sky objects.
    maxiters: positive int, time of iterations.
    showmedian: boolean. If true, the median of sky objects values will be returned instead of the average.

    Returns:
    -----------
    mean/median value of nearby sky objects.
    '''
    from astropy.coordinates import match_coordinates_sky
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clip

    ra, dec = cen_ra, cen_dec
    bkg_pos = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    catalog = SkyCoord(ra=sky_cat['i_ra'] * u.degree, dec=sky_cat['i_dec'] * u.degree)
    if type(matching_radius) == list:
        if len(matching_radius) != 2:
            raise SyntaxError('The length of matching_radius list must be 2!')
        else:
            obj_inx1 = np.where(catalog.separation(bkg_pos) < matching_radius[1] * u.arcmin)[0]
            obj_inx2 = np.where(catalog.separation(bkg_pos) > matching_radius[0] * u.arcmin)[0]
            obj_inx = np.intersect1d(obj_inx1, obj_inx2)
    else:
        obj_inx = np.where(catalog.separation(bkg_pos) < matching_radius * u.arcmin)[0]
    if print_number:
        print('Sky objects number around' + str(matching_radius) + 'arcmin: ', len(obj_inx))


    x = sky_cat[obj_inx]['r_apertureflux_' + aperture +'_flux'] * 1.7378e30 / (np.pi * SkyObj_aperture_dic[aperture]**2)
    x = sigma_clip(x, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=maxiters)
    x = x.data[~x.mask]

    if showmedian:
        return np.nanmedian(x)
    else:
        return np.nanmean(x)


"""
# Evaluate the mean sky value
def evaluate_sky(img, sigma=1.5, radius=15, threshold=0.005, deblend_cont=0.001, deblend_nthresh=20, clean_param=1.0, show_fig=True, show_hist=True):
    '''Evaluate the mean sky value.
    Parameters:
    ----------
    img: 2-D numpy array, the input image
    show_fig: bool. If True, it will show you the masked sky image.
    show_hist: bool. If True, it will show you the histogram of the sky value.
    
    Returns:
    -------
    bkg_global: `sep` object.
    '''
    b = 50  # Box size
    f = 5   # Filter width

    bkg = sep.Background(img, maskthresh=0, bw=b, bh=b, fw=f, fh=f)

    obj_lthre, seg_lthre = sep.extract(img, sigma,
                                       err=bkg.globalrms, 
                                       minarea=20, 
                                       deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                                       clean=True, clean_param=clean_param,
                                       segmentation_map=True)

    seg_sky = copy.deepcopy(seg_lthre)
    seg_sky[seg_lthre > 0] = 1

    # Convolve the image with a Gaussian kernel with the width of 15 pixel
    bkg_mask = convolve(seg_sky.astype('float'), Gaussian2DKernel(radius))
    bkg_mask = bkg_mask >= threshold

    if show_fig:
        display_single((~bkg_mask)*(img - bkg.globalback),
                   scale_bar=False)
    if show_hist:
        from scipy import stats
        samp = img[~bkg_mask]
        x = np.linspace(-0.5, 0.5, 100)
        gkde = stats.gaussian_kde(dataset=samp)
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(x, gkde.evaluate(x), linestyle='dashed', c='black', lw=2,
                label='KDE')
        ax.hist(samp, bins=x, normed=1)
        ax.legend(loc='best', frameon=False, fontsize=20)


        ax.set_title('Histogram of pixels', fontsize=20)
        ax.set_xlabel('Pixel Value', fontsize=20)
        ax.set_ylabel('Normed Number', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.set_ylim(0,20)
        offset = x[np.argmax(gkde.evaluate(x))]
        ax.text(-0.045, 10, r'$\mathrm{offset}='+str(round(offset, 6))+'$', fontsize=20)
        plt.vlines(np.median(samp), 0, 20, linestyle='--')
        print('mean', np.mean(samp))

    bkg_global = sep.Background(img, 
                                mask=bkg_mask, maskthresh=0,
                                bw=b, bh=b, 
                                fw=f, fh=f)
    print("# Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_global.globalback, bkg_global.globalrms))
    return bkg_global
"""

"""
# Evaluate the median sky value: new edition, still using convolution
def evaluate_sky(img, sigma=1.5, radius=10, pixel_scale=0.168, central_mask_radius=7.0, 
                 threshold=0.005, deblend_cont=0.001, deblend_nthresh=20, 
                 clean_param=1.0, show_fig=True, show_hist=True):
    '''Evaluate the mean sky value.
    Parameters:
    ----------
    img: 2-D numpy array, the input image
    show_fig: bool. If True, it will show you the masked sky image.
    show_hist: bool. If True, it will show you the histogram of the sky value.
    
    Returns:
    -------
    median: median of background pixels, in original unit
    std: standard deviation, in original unit
    '''
    import sep
    import copy 
    from .imutils import extract_obj, make_binary_mask
    from astropy.convolution import convolve, Gaussian2DKernel
    b = 50  # Box size
    f = 5   # Filter width

    bkg = sep.Background(img, maskthresh=0, bw=b, bh=b, fw=f, fh=f)
    # first time
    objects, segmap = extract_obj(img, b=30, f=5, sigma=sigma,
                                       minarea=20, pixel_scale=pixel_scale,
                                       deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                                       clean_param=clean_param, show_fig=False)
    seg_sky = copy.deepcopy(segmap)
    seg_sky[segmap > 0] = 1
    # Convolve the image with a Gaussian kernel with the width of 15 pixel
    bkg_mask = convolve(seg_sky.astype('float'), Gaussian2DKernel(2 * radius))
    bkg_mask_1 = (bkg_mask >= threshold)
    
    data = copy.deepcopy(img)
    data[bkg_mask_1 == 1] = 0
    
    # Second time
    obj_lthre, seg_lthre = extract_obj(data, b=30, f=5, sigma=sigma + 1,
                                       minarea=5, pixel_scale=pixel_scale,
                                       deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                                       clean_param=clean_param, show_fig=False)
    seg_sky = copy.deepcopy(seg_lthre)
    seg_sky[seg_lthre > 0] = 1
    # Convolve the image with a Gaussian kernel with the width of 15 pixel
    bkg_mask = convolve(seg_sky.astype('float'), Gaussian2DKernel(radius))
    bkg_mask_2 = (bkg_mask >= threshold / 5)
    
    bkg_mask = (bkg_mask_1 + bkg_mask_2).astype(bool)
    
    cen_obj = objects[segmap[int(bkg_mask.shape[0] / 2.), int(bkg_mask.shape[1] / 2.)] - 1]
    fraction_radius = sep.flux_radius(img, cen_obj['x'], cen_obj['y'], 10*cen_obj['a'], 0.5)[0]
    print(fraction_radius)
    ba = np.divide(cen_obj['b'], cen_obj['a'])
    if fraction_radius < int(bkg_mask.shape[0] / 5.):
        sep.mask_ellipse(bkg_mask, cen_obj['x'], cen_obj['y'], fraction_radius, fraction_radius * ba,
                        cen_obj['theta'], r=central_mask_radius)
    elif fraction_radius < int(bkg_mask.shape[0] / 3.):
        sep.mask_ellipse(bkg_mask, cen_obj['x'], cen_obj['y'], fraction_radius, fraction_radius * ba,
                        cen_obj['theta'], r=0.5)

    bkg_global = sep.Background(img, 
                                mask=bkg_mask, maskthresh=0,
                                bw=30, bh=30, 
                                fw=5, fh=5)
    print("# Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_global.globalback, bkg_global.globalrms))
    
    # Estimate sky from histogram of binned image
    import copy
    from scipy import stats
    from astropy.stats import sigma_clip
    from astropy.nddata import block_reduce
    data = copy.deepcopy(img)
    data[bkg_mask] = np.nan
    f_factor = round(8 / pixel_scale)
    rebin = block_reduce(data, f_factor)
    sample = rebin.flatten()
    if show_fig:
        display_single(rebin)
        plt.savefig('./{}-bkg.png'.format(np.random.randint(1000)), dpi=100, bbox_inches='tight')
    
    temp = sigma_clip(sample)
    sample = temp.data[~temp.mask]
    #sample = sample[~np.isnan(sample)]

    kde = stats.gaussian_kde(sample)
    
    mean = np.mean(sample) / f_factor**2
    median = np.median(sample) / f_factor**2
    std = np.std(sample, ddof=1) / f_factor

    xlim = np.std(sample, ddof=1) * 7
    x = np.linspace(-xlim, xlim, 100)
    offset = x[np.argmax(kde.evaluate(x))] / f_factor**2

    print('mean', mean)
    print('median', median)
    print('std', std)
    
    if show_hist:
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(x, kde.evaluate(x), linestyle='dashed', c='black', lw=2,
                label='KDE')
        ax.hist(sample, bins=x, normed=1);
        ax.legend(loc='best', frameon=False, fontsize=20)

        ax.set_xlabel('Pixel Value', fontsize=20)
        ax.set_ylabel('Normed Number', fontsize=20)
        ax.tick_params(labelsize=20)
        ylim = ax.get_ylim()
        ax.text(-0.1 * f_factor, 0.9 * (ylim[1] - ylim[0]) + ylim[0], 
                r'$\mathrm{offset}='+str(round(offset, 6))+'$', fontsize=20)
        ax.text(-0.1 * f_factor, 0.8 * (ylim[1] - ylim[0]) + ylim[0],
                r'$\mathrm{median}='+str(round(median, 6))+'$', fontsize=20)
        ax.text(-0.1 * f_factor, 0.7 * (ylim[1] - ylim[0]) + ylim[0],
                r'$\mathrm{std}='+str(round(std, 6))+'$', fontsize=20)
        plt.vlines(np.median(sample), 0, ylim[1], linestyle='--')

    return median, std
"""

# Evaluate the median sky value: new edition, not using convolution
def evaluate_sky(img, sigma=1.5, radius=10, pixel_scale=0.168, central_mask_radius=7.0, 
                 threshold=0.005, deblend_cont=0.001, deblend_nthresh=20, 
                 clean_param=1.0, show_fig=True, show_hist=True, f_factor=None):
    '''Evaluate the mean sky value.
    Parameters:
    ----------
    img: 2-D numpy array, the input image
    show_fig: bool. If True, it will show you the masked sky image.
    show_hist: bool. If True, it will show you the histogram of the sky value.
    
    Returns:
    -------
    median: median of background pixels, in original unit
    std: standard deviation, in original unit
    '''
    import sep
    import copy 
    from slug.imutils import extract_obj, make_binary_mask
    from astropy.convolution import convolve, Gaussian2DKernel
    b = 50  # Box size
    f = 5   # Filter width

    bkg = sep.Background(img, maskthresh=0, bw=b, bh=b, fw=f, fh=f)
    # first time
    objects, segmap = extract_obj(img - bkg.globalback, b=30, f=5, sigma=sigma,
                                    minarea=20, pixel_scale=pixel_scale,
                                    deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                                    clean_param=clean_param, show_fig=False)
    
    seg_sky = copy.deepcopy(segmap)
    seg_sky[segmap > 0] = 1
    seg_sky = seg_sky.astype(bool)
    # Blow up the mask
    for obj in objects:
        sep.mask_ellipse(seg_sky, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], r=radius)
    bkg_mask_1 = seg_sky
    
    data = copy.deepcopy(img - bkg.globalback)
    data[bkg_mask_1 == 1] = 0

    # Second time
    obj_lthre, seg_lthre = extract_obj(data, b=30, f=5, sigma=sigma + 1,
                                       minarea=5, pixel_scale=pixel_scale,
                                       deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont,
                                       clean_param=clean_param, show_fig=False)
    seg_sky = copy.deepcopy(seg_lthre)
    seg_sky[seg_lthre > 0] = 1
    seg_sky = seg_sky.astype(bool)
    # Blow up the mask
    for obj in obj_lthre:
        sep.mask_ellipse(seg_sky, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], r=radius/2)
    bkg_mask_2 = seg_sky
    
    bkg_mask = (bkg_mask_1 + bkg_mask_2).astype(bool)
    
    cen_obj = objects[segmap[int(bkg_mask.shape[0] / 2.), int(bkg_mask.shape[1] / 2.)] - 1]
    fraction_radius = sep.flux_radius(img, cen_obj['x'], cen_obj['y'], 10*cen_obj['a'], 0.5)[0]
    
    ba = np.divide(cen_obj['b'], cen_obj['a'])
    
    if fraction_radius < int(bkg_mask.shape[0] / 8.):
        sep.mask_ellipse(bkg_mask, cen_obj['x'], cen_obj['y'], fraction_radius, fraction_radius * ba,
                        cen_obj['theta'], r=central_mask_radius)
    elif fraction_radius < int(bkg_mask.shape[0] / 4.):
        sep.mask_ellipse(bkg_mask, cen_obj['x'], cen_obj['y'], fraction_radius, fraction_radius * ba,
                        cen_obj['theta'], r=1.2)
    
    # Estimate sky from histogram of binned image
    import copy
    from scipy import stats
    from astropy.stats import sigma_clip
    from astropy.nddata import block_reduce
    data = copy.deepcopy(img)
    data[bkg_mask] = np.nan
    if f_factor is None:
        f_factor = round(6 / pixel_scale)
    rebin = block_reduce(data, f_factor)
    sample = rebin.flatten()
    if show_fig:
        display_single(rebin)
        plt.savefig('./{}-bkg.png'.format(np.random.randint(1000)), dpi=100, bbox_inches='tight')
    
    temp = sigma_clip(sample)
    sample = temp.data[~temp.mask]

    kde = stats.gaussian_kde(sample)
    
    mean = np.mean(sample) / f_factor**2
    median = np.median(sample) / f_factor**2
    std = np.std(sample, ddof=1) / f_factor / np.sqrt(len(sample))

    xlim = np.std(sample, ddof=1) * 7
    x = np.linspace(-xlim + np.median(sample), xlim + np.median(sample), 100)
    offset = x[np.argmax(kde.evaluate(x))] / f_factor**2
    
    print('mean', mean)
    print('median', median)
    print('std', std)

    bkg_global = sep.Background(img, 
                                mask=bkg_mask, maskthresh=0,
                                bw=f_factor, bh=f_factor, 
                                fw=f_factor/2, fh=f_factor/2)
    print("#SEP sky: Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_global.globalback, bkg_global.globalrms))

    if show_hist:
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(x, kde.evaluate(x), linestyle='dashed', c='black', lw=2,
                label='KDE')
        ax.hist(sample, bins=x, normed=1);
        ax.legend(loc='best', frameon=False, fontsize=20)

        ax.set_xlabel('Pixel Value', fontsize=20)
        ax.set_ylabel('Normed Number', fontsize=20)
        ax.tick_params(labelsize=20)
        ylim = ax.get_ylim()
        ax.text(-0.1 * f_factor + np.median(sample), 0.9 * (ylim[1] - ylim[0]) + ylim[0], 
                r'$\mathrm{offset}='+str(round(offset, 6))+'$', fontsize=20)
        ax.text(-0.1 * f_factor + np.median(sample), 0.8 * (ylim[1] - ylim[0]) + ylim[0],
                r'$\mathrm{median}='+str(round(median, 6))+'$', fontsize=20)
        ax.text(-0.1 * f_factor + np.median(sample), 0.7 * (ylim[1] - ylim[0]) + ylim[0],
                r'$\mathrm{std}='+str(round(std, 6))+'$', fontsize=20)
        plt.vlines(np.median(sample), 0, ylim[1], linestyle='--')

    return median, std


# Evaluate the mean sky value for Dragonfly
def evaluate_sky_dragonfly(img, b=15, f=3, sigma=1.5, radius=1.0, threshold=0.05, show_fig=True, show_hist=True):
    '''Evaluate the mean sky value.
    Parameters:
    ----------
    img: 2-D numpy array, the input image
    show_fig: bool. If True, it will show you the masked sky image.
    show_hist: bool. If True, it will show you the histogram of the sky value.
    
    Returns:
    -------
    bkg_global: `sep` object.
    '''
    b = b  # Box size
    f = f   # Filter width


    bkg = sep.Background(img, maskthresh=0, bw=b, bh=b, fw=f, fh=f)

    obj_lthre, seg_lthre = extract_obj(img, b=b, f=f, sigma=sigma, pixel_scale=Dragonfly_pixel_scale,
                                            deblend_nthresh=128, deblend_cont=0.0001, show_fig=show_fig)

    # make mask
    seg_mask = make_binary_mask(img, None, seg_lthre, radius=2.0, show_fig=show_fig, threshold=0.005, gaia=False)

    if show_hist:
        from scipy import stats
        samp = img[~bkg_mask]
        x = np.linspace(-0.5, 0.5, 100)
        gkde = stats.gaussian_kde(dataset=samp)
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(x, gkde.evaluate(x), linestyle='dashed', c='black', lw=2,
                label='KDE')
        ax.hist(samp, bins=x, normed=1)
        ax.legend(loc='best', frameon=False, fontsize=20)


        ax.set_title('Histogram of pixels', fontsize=20)
        ax.set_xlabel('Pixel Value', fontsize=20)
        ax.set_ylabel('Normed Number', fontsize=20)
        ax.tick_params(labelsize=20)
        ax.set_ylim(0,20)
        offset = x[np.argmax(gkde.evaluate(x))]
        ax.text(-0.045, 10, r'$\mathrm{offset}='+str(round(offset, 6))+'$', fontsize=20)
        plt.vlines(np.median(samp), 0, 20, linestyle='--')
        print('mean', np.mean(samp))

    bkg_global = sep.Background(img, 
                                mask=seg_mask, maskthresh=0,
                                bw=20, bh=20, 
                                fw=5, fh=5)
    print("# Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_global.globalback, bkg_global.globalrms))
    return bkg_global


# Run surface brightness profile for the given image and mask
def run_SBP(img_path, msk_path, pixel_scale, phys_size, iraf_path, step=0.10, 
    sma_ini=10.0, sma_max=900.0, n_clip=3, maxTry=5, low_clip=3.0, upp_clip=2.5, force_e=None, r_interval=(20, 50), outPre=None):
    # Centeral coordinate 
    img_data = fits.open(img_path)[0].data
    x_cen, y_cen = int(img_data.shape[0]/2), int(img_data.shape[1]/2)

    # Initial guess of axis ratio and position angle 
    ba_ini, pa_ini = 0.5, 90.0

    # Initial radius of Ellipse fitting
    sma_ini = sma_ini

    # Minimum and maximum radiuse of Ellipse fitting
    sma_min, sma_max = 0.0, sma_max

    # Stepsize of Ellipse fitting. By default we are not using linear step size
    step = step

    # Behaviour of Ellipse fitting
    stage = 2   # Fix the central coordinate of every isophote at the x_cen / y_cen position

    # Pixel scale of the image.
    pix_scale = pixel_scale

    # Photometric zeropoint 
    zeropoint = 27.0

    # Exposure time
    exptime = 1.0

    # Along each isophote, Elipse will perform sigmal clipping to remove problematic pixels
    # The behaviour is decided by these three parameters: Number of sigma cliping, lower, and upper clipping threshold 
    n_clip, low_clip, upp_clip = n_clip, low_clip, upp_clip

    # After the clipping, Ellipse can use the mean, median, or bi-linear interpolation of the remain pixel values
    # as the "average intensity" of that isophote 
    integrade_mode = 'median'   # or 'mean', or 'bi-linear'

    ISO = iraf_path + 'x_isophote.e'
    # This is where `x_isophote.e` exists

    TBL = iraf_path + 'x_ttools.e'
    # This is where `x_ttools.e` exists. Actually it's no need to install IRAF at all.

    # Make 'Data' to save your output data
    if not os.path.isdir('Data'):
        os.mkdir('Data')

    # Start running Ellipse
    ell_2, bin_2 = galSBP.galSBP(img_path, 
                                 mask=msk_path,
                                 galX=x_cen, galY=y_cen,
                                 galQ=ba_ini, galPA=pa_ini,
                                 iniSma=sma_ini, 
                                 minSma=sma_min, maxSma=sma_max,
                                 pix=1/pix_scale, zpPhoto=zeropoint,
                                 expTime=exptime, 
                                 stage=stage,
                                 ellipStep=step,
                                 isophote=ISO, 
                                 xttools=TBL, 
                                 uppClip=upp_clip, lowClip=low_clip, 
                                 nClip=n_clip, 
                                 maxTry=maxTry,
                                 fracBad=0.8,
                                 maxIt=300,
                                 harmonics="none",
                                 intMode=integrade_mode, 
                                 saveOut=True, plMask=True,
                                 verbose=True, savePng=False, 
                                 updateIntens=False, saveCsv=True,
                                 suffix='', location='./Data/', outPre=outPre+'-ellip-2')

    # Calculate the mean ellipticity and position angle in 20 kpc ~ 50 kpc
    interval = np.intersect1d(np.where(ell_2['sma'].data*pixel_scale*phys_size > r_interval[0]),
               np.where(ell_2['sma'].data*pixel_scale*phys_size < r_interval[1]))
    mean_e = ell_2['ell'][interval].mean()
    stdev_e = ell_2['ell'][interval].std()
    mean_pa = ell_2['pa_norm'][interval].mean()
    stdev_pa = ell_2['pa_norm'][interval].std()

    print ('\n')
    print ('mean ellipticity:', mean_e)
    print ('std ellipticity:', stdev_e)
    print ('mean pa:', mean_pa)
    print ('std pa:', stdev_pa)
    print ('\n')
    # RUN Ellipse for the second time, fixing shape and center
    stage = 3

    # Initial value of axis ratio and position angle, based on previous fitting
    if force_e is not None:
        ba_ini = 1 - force_e
    else:
        ba_ini = 1 - mean_e
    pa_ini = mean_pa

    step = 0.1

    ell_3, bin_3 = galSBP.galSBP(img_path, 
                                 mask=msk_path,
                                 galX=x_cen, galY=y_cen,
                                 galQ=ba_ini, galPA=pa_ini,
                                 iniSma=sma_ini, 
                                 minSma=sma_min, maxSma=sma_max,
                                 pix=1/pixel_scale, zpPhoto=zeropoint,
                                 expTime=exptime, 
                                 stage=stage,
                                 ellipStep=step,
                                 isophote=ISO, 
                                 xttools=TBL, 
                                 uppClip=upp_clip, lowClip=low_clip, 
                                 nClip=n_clip, 
                                 maxTry=5,
                                 fracBad=0.8,
                                 maxIt=300,
                                 harmonics="none",
                                 intMode=integrade_mode, 
                                 saveOut=True, plMask=True,
                                 verbose=True, savePng=False, 
                                 updateIntens=False, saveCsv=True,
                                 suffix='', location='./Data/', outPre=outPre+'-ellip-3')
    return ell_2, ell_3
