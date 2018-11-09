# Import packages
import os
import copy

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

import sep

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

# Calculate physical size of a given redshift
def phys_size(redshift):
    '''Calculate the corresponding physical size per arcsec of a given redshift.
    Requirement:
    -----------
    cosmology: https://github.com/esheldon/cosmology
    
    Parameters:
    -----------
    redshift: float
    
    Returns:
    -----------
    physical_size: float, in 'kpc/arcsec'
    '''
    import cosmology
    cosmos = cosmology.Cosmo(H0=70, omega_m=0.3, flat=True, omega_l=0.7, omega_k=None)
    ang_distance = cosmos.Da(0.0, redshift)
    physical_size = ang_distance/206265*1000 # kpc/arcsec
    print 'At this redshift, 1 arcsec =', physical_size, 'kpc'
    return physical_size

# Rebin a image / mask
def rebin(array, dimensions=None, scale=None):
    """ From http://martynbristow.co.uk/wordpress/blog/rebinning-data/
        It's a little bit slow, but flux is conserved
        Return the array ``array`` to the new ``dimensions`` conserving flux the flux in the bins
        The sum of the array will remain the same
        
        >>> ar = numpy.array([
        [0,1,2],
        [1,2,3],
        [2,3,4]
        ])
        >>> rebin(ar, (2,2))
        array([
        [1.5, 4.5]
        [4.5, 7.5]
        ])
        Raises
        ------
        
        AssertionError
        If the totals of the input and result array don't agree, raise an error because computation may have gone wrong
        
        Reference
        =========
        +-+-+-+
        |1|2|3|
        +-+-+-+
        |4|5|6|
        +-+-+-+
        |7|8|9|
        +-+-+-+
        """
    import numpy as np
    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(array.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(array.shape)
        elif len(dimensions) != len(array.shape):
            raise RuntimeError('')
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = map(int, map(round, map(lambda x: x*scale, array.shape)))
        elif len(scale) != len(array.shape):
            raise RuntimeError('')
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\trebin(array, dimensions=(x,y))\n\trebin(array, scale=a')
    print dimensions
    print "Rebinning to Dimensions: %s, %s" % tuple(dimensions)
    import itertools
    dY, dX = map(divmod, map(float, array.shape), dimensions)

    result = np.zeros(dimensions)
    for j, i in itertools.product(*map(xrange, array.shape)):
        (J, dj), (I, di) = divmod(j*dimensions[0], array.shape[0]), divmod(i*dimensions[1], array.shape[1])
        (J1, dj1), (I1, di1) = divmod(j+1, array.shape[0]/float(dimensions[0])), divmod(i+1, array.shape[1]/float(dimensions[1]))
    
        # Moving to new bin
        # Is this a discrete bin?
        dx,dy=0,0
        if (I1-I == 0) | ((I1-I == 1) & (di1==0)):
            dx = 1
        else:
            dx=1-di1
        if (J1-J == 0) | ((J1-J == 1) & (dj1==0)):
            dy=1
        else:
            dy=1-dj1
        # Prevent it from allocating outide the array
        I_=min(dimensions[1]-1,I+1)
        J_=min(dimensions[0]-1,J+1)
        result[J, I] += array[j,i]*dx*dy
        result[J_, I] += array[j,i]*(1-dy)*dx
        result[J, I_] += array[j,i]*dy*(1-dx)
        result[J_, I_] += array[j,i]*(1-dx)*(1-dy)
    allowError = 0.1
    assert (array.sum() < result.sum() * (1+allowError)) & (array.sum() >result.sum() * (1-allowError))
    return result

# Extract objects for a given image
def extract_obj(img, b=30, f=5, sigma=5, show_fig=True, pixel_scale=0.168, minarea=5, deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0):
    '''Extract objects for a given image, using `sep`.

    Parameters:
    ----------
    img: 2-D numpy array
    b: float, size of box
    f: float, size of convolving kernel
    sigma: float, detection threshold
    pixel_scale: float

    Returns:
    -------
    objects: numpy array, containing the positions and shapes of extracted objects.
    segmap: 2-D numpy array, segmentation map
    '''
    # Subtract a mean sky value to achieve better object detection
    b = 30  # Box size
    f = 5   # Filter width
    bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    data_sub = img - bkg.back()

    sigma = sigma
    objects, segmap = sep.extract(data_sub,
                                  sigma,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea)
                                  
    print("# Detect %d objects" % len(objects))

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0] = display_single(data_sub, ax=ax[0], scale_bar_length=60, pixel_scale=pixel_scale)

        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=8*obj['a'],
                        height=8*obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP , ax=ax[1])
    return objects, segmap

# Make binary mask
def make_binary_mask(img, w, segmap, radius=10.0, threshold=0.01, gaia=True, factor_b=1.3):
    '''Make binary mask for a given segmentation map. 
    We convolve the segmentation map using a Gaussian kernal to expand the size of mask.

    Parameters:
    ----------
    img: 2-D numpy array, image data
    w: wcs of the input image
    segmap: 2-D numpy array, segmentation map given by `extract_obj()`
    radius: float, the width of Gaussian kernel
    threshold: float, it can change the size of mask. Lower threshold, larger mask.

    Returns:
    -------
    binary_mask: 2-D numpy boolean array.
    '''

    # Remove the central object
    seg_nocen = imtools.seg_remove_cen_obj(segmap)
    seg_conv = copy.deepcopy(seg_nocen)
    seg_conv[seg_nocen > 0] = 1

    # Convolve the image with a Gaussian kernel with the width of 10 pixel
    # This is actually pretty slow, because the image is very large. 
    seg_conv = convolve(seg_conv.astype('float'), Gaussian2DKernel(radius))
    seg_mask = seg_conv >= threshold

    # Combine this mask with Gaia star mask
    gaia_mask = imtools.gaia_star_mask(img, w, gaia_bright=16, factor_f=10000, factor_b=factor_b)[1].astype('bool')

    display_single((seg_mask + gaia_mask).astype(int), cmap=SEG_CMAP)

    binary_mask = seg_mask + gaia_mask
    return binary_mask

# Evaluate the mean sky value
def evaluate_sky(img, show_fig=True, show_hist=True):
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
    b = 30  # Box size
    f = 5   # Filter width
    bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)

    obj_lthre, seg_lthre = sep.extract(img, 2.0,
                                       err=bkg.globalrms, 
                                       minarea=20, 
                                       deblend_nthresh=20, deblend_cont=0.1,
                                       clean=True, clean_param=1.0,
                                       segmentation_map=True)

    seg_sky = copy.deepcopy(seg_lthre)
    seg_sky[seg_lthre > 0] = 1

    # Convolve the image with a Gaussian kernel with the width of 15 pixel
    bkg_mask = convolve(seg_sky.astype('float'), Gaussian2DKernel(15.0))
    bkg_mask = bkg_mask >= 0.005

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
                                bw=100, bh=100, 
                                fw=20, fh=20)
    print("# Mean Sky / RMS Sky = %10.5f / %10.5f" % (bkg_global.globalback, bkg_global.globalrms))
    return bkg_global


def run_SBP(img_path, msk_path, pixel_scale, phys_size, iraf_path, step=0.10, n_clip=3, low_clip=3.0, upp_clip=2.5, force_e=None):
    # Centeral coordinate 
    img_data = fits.open(img_path)[0].data
    cen = img_data.shape[0]/2
    x_cen, y_cen = cen, cen

    # Initial guess of axis ratio and position angle 
    ba_ini, pa_ini = 0.5, 90.0

    # Initial radius of Ellipse fitting
    sma_ini = 10.0

    # Minimum and maximum radiuse of Ellipse fitting
    sma_min, sma_max = 0.0, 2000.0

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
                                 maxTry=3,
                                 fracBad=0.8,
                                 maxIt=300,
                                 harmonics="none",
                                 intMode=integrade_mode, 
                                 saveOut=True, plMask=True,
                                 verbose=True, savePng=False, 
                                 updateIntens=False, saveCsv=True,
                                 suffix='', location='./Data/')

    # Calculate the mean ellipticity and position angle in 20 kpc ~ 50 kpc
    interval = np.intersect1d(np.where(ell_2['sma'].data*pixel_scale*phys_size > 20),
               np.where(ell_2['sma'].data*pixel_scale*phys_size < 50))
    mean_e = ell_2['ell'][interval].mean()
    stdev_e = ell_2['ell'][interval].std()
    mean_pa = ell_2['pa_norm'][interval].mean()
    stdev_pa = ell_2['pa_norm'][interval].std()

    print 'mean ellipticity:', mean_e
    print 'std ellipticity:', stdev_e
    print 'mean pa:', mean_pa
    print 'std pa:', stdev_pa

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
                                 intMode=integrade_mode, 
                                 saveOut=True, plMask=True,
                                 verbose=True, savePng=False, 
                                 updateIntens=False, saveCsv=True,
                                 suffix='', location='./Data/')
    return ell_2, ell_3