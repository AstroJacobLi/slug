# Import packages
import os
import copy
import slug

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.table import Table

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

# Generate DECaLS image url
def gen_url_decals(ra, dec, size, bands, layer='decals-dr7', pixel_unit=False):
    '''Generate image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    size: float, image size (pixel)
    bands: string, such as 'r' or 'gri'
    
    Returns:
    -----------
    url: list of str, url of S18A image.  
    '''

    if pixel_unit:
        return ['http://legacysurvey.org/viewer/fits-cutout?ra='
            + str(ra)
            + '&dec='
            + str(dec)
            + '&pixscale='
            + str(DECaLS_pixel_scale)
            + '&layer='
            + layer
            + '&size='
            + str(size)
            + '&bands='
            + bands]
    else:        
        return ['http://legacysurvey.org/viewer/fits-cutout?ra='
            + str(ra)
            + '&dec='
            + str(dec)
            + '&pixscale='
            + str(DECaLS_pixel_scale)
            + '&layer='
            + layer
            + '&size='
            + str(size/DECaLS_pixel_scale)
            + '&bands='
            + bands]

# Login NAOJ server
def login_naoj_server(config_path):
    ''' Runs well under python 2. In python 3, there's a widget 
    to enter username and password directly in Jupyter Notebook.'''
    import urllib
    import urllib2
    # Import HSC username and password
    config = Table.read(config_path, format='ascii.no_header')['col1']
    username = config[0]
    password = config[1]
    # Create a password manager
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

    # Add the username and password.
    top_level_url = 'https://hscdata.mtk.nao.ac.jp/das_quarry/dr2.1/'
    password_mgr.add_password(None, top_level_url, username, password)
    handler = urllib2.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(handler)

    # use the opener to fetch a URL
    opener.open(top_level_url)

    # Install the opener.
    # Now all calls to urllib2.urlopen use our opener.
    urllib2.install_opener(opener)

# Generate HSC image url
def gen_url_hsc_s18a(ra, dec, w, h, band, pixel_unit=False):
    '''Generate image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    w: float, width (arcsec)
    h: float, height (arcsec)
    band: string, such as 'r'
    
    Returns:
    -----------
    url: list of str, url of S18A image.  
    '''
    if pixel_unit:
        return ['https://hscdata.mtk.nao.ac.jp/das_quarry/dr2.1/cgi-bin/cutout?ra='
            + str(ra) 
            + '&dec='
            + str(dec)
            + '&sw='
            + str(w*HSC_pixel_scale)
            + 'asec&sh='
            + str(h*HSC_pixel_scale)
            + 'asec&type=coadd&image=on&variance=on&filter=HSC-'
            + str(band.upper())
            + '&tract=&rerun=s18a_wide']
    else:        
        return ['https://hscdata.mtk.nao.ac.jp/das_quarry/dr2.1/cgi-bin/cutout?ra='
           + str(ra) 
           + '&dec='
           + str(dec)
           + '&sw='
           + str(w)
           + 'asec&sh='
           + str(h)
           + 'asec&type=coadd&image=on&variance=on&filter=HSC-'
           + str(band.upper())
           + '&tract=&rerun=s18a_wide']

def gen_url_hsc_s16a(ra, dec, w, h, band, pixel_unit=False):
    '''Generate image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    w: float, width (arcsec)
    h: float, height (arcsec)
    band: string, such as 'r'

    Returns:
    -----------
    url: str, url of S16A image.  
    '''
    if pixel_unit:
        return ['https://hscdata.mtk.nao.ac.jp/das_quarry/dr1/cgi-bin/quarryImage?ra='
            + str(ra) 
            + '&dec='
            + str(dec)
            + '&sw='
            + str(w*HSC_pixel_scale)
            + 'asec&sh='
            + str(h*HSC_pixel_scale)
            + 'asec&type=coadd&image=on&variance=on&filter=HSC-'
            + str(band.upper())
            + '&tract=&rerun=s16a_wide2']
    else:
        return ['https://hscdata.mtk.nao.ac.jp/das_quarry/dr1/cgi-bin/quarryImage?ra='
           + str(ra) 
           + '&dec='
           + str(dec)
           + '&sw='
           + str(w)
           + 'asec&sh='
           + str(h)
           + 'asec&type=coadd&image=on&variance=on&filter=HSC-'
           + str(band.upper())
           + '&tract=&rerun=s16a_wide2']


# Calculate physical size of a given redshift
def phys_size(redshift, is_print=True, H0=70, Omegam=0.3, Omegal=0.7):
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
    cosmos = cosmology.Cosmo(H0=H0, omega_m=Omegam, flat=True, omega_l=Omegal, omega_k=None)
    ang_distance = cosmos.Da(0.0, redshift)
    physical_size = ang_distance/206265*1000 # kpc/arcsec
    if is_print:
    	print ('At redshift', redshift, ', 1 arcsec =', physical_size, 'kpc')
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
    print (dimensions)
    print ("Rebinning to Dimensions: %s, %s" % tuple(dimensions))
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
def make_binary_mask(img, w, segmap, radius=10.0, threshold=0.01, gaia=True, factor_b=1.2, show_fig=True):
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

    if gaia is False:
        if show_fig:
            display_single(seg_mask.astype(int), cmap=SEG_CMAP)
        return seg_mask
    else:
        # Combine this mask with Gaia star mask
        gaia_mask = imtools.gaia_star_mask(img, w, gaia_bright=16, factor_f=10000, factor_b=factor_b)[1].astype('bool')
        if show_fig:
        	display_single((seg_mask + gaia_mask).astype(int), cmap=SEG_CMAP)

        binary_mask = seg_mask + gaia_mask
        return binary_mask

# Evaluate the mean sky value
def evaluate_sky(img, sigma=1.5, radius=15, threshold=0.005, show_fig=True, show_hist=True):
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
                                       deblend_nthresh=20, deblend_cont=0.001,
                                       clean=True, clean_param=1.0,
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

    obj_lthre, seg_lthre = slug.extract_obj(img, b=b, f=f, sigma=sigma, pixel_scale=slug.Dragonfly_pixel_scale,
                                            deblend_nthresh=128, deblend_cont=0.0001, show_fig=show_fig)

    # make mask
    seg_mask = slug.make_binary_mask(img, None, seg_lthre, radius=2.0, show_fig=show_fig, threshold=0.005, gaia=False)

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

def run_SBP(img_path, msk_path, pixel_scale, phys_size, iraf_path, step=0.10, sma_ini=10.0, sma_max=900.0, n_clip=3, low_clip=3.0, upp_clip=2.5, force_e=None, outPre=None):
    # Centeral coordinate 
    img_data = fits.open(img_path)[0].data
    cen = img_data.shape[0]/2
    x_cen, y_cen = cen, cen

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
                                 maxTry=5,
                                 fracBad=0.8,
                                 maxIt=300,
                                 harmonics="none",
                                 intMode=integrade_mode, 
                                 saveOut=True, plMask=True,
                                 verbose=True, savePng=False, 
                                 updateIntens=False, saveCsv=True,
                                 suffix='', location='./Data/', outPre=outPre+'-ellip-2')

    # Calculate the mean ellipticity and position angle in 20 kpc ~ 50 kpc
    interval = np.intersect1d(np.where(ell_2['sma'].data*pixel_scale*phys_size > 20),
               np.where(ell_2['sma'].data*pixel_scale*phys_size < 50))
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

    step = 0.2

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

def display_isophote(img, ell, pixel_scale, scale_bar=True, scale_bar_length=50, physical_scale=None, text=None, ax=None, contrast=None, circle=None):
    """Visualize the isophotes."""
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)

        # Whole central galaxy: Step 2
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax
        
    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_major_formatter(NullFormatter())

    cen = img.shape[0]/2

    if contrast is not None:
        ax1 = display_single(img, pixel_scale=pixel_scale, ax=ax1, scale_bar=scale_bar, scale_bar_length=scale_bar_length,
                    physical_scale=physical_scale, contrast=contrast, add_text=text)
    else:
        ax1 = display_single(img, pixel_scale=pixel_scale, ax=ax1, scale_bar=scale_bar, scale_bar_length=scale_bar_length,
                    physical_scale=physical_scale, contrast=0.25, add_text=text)
    
    for k, iso in enumerate(ell):
        if k % 2 == 0:
            e = Ellipse(xy=(iso['x0'], iso['y0']),
                        height=iso['sma'] * 2.0,
                        width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                        angle=iso['pa'])
            e.set_facecolor('none')
            e.set_edgecolor('r')
            e.set_alpha(0.5)
            e.set_linewidth(1.5)
            ax1.add_artist(e)
    ax1.set_aspect('equal')

    if circle is not None:
        if physical_scale is not None:
            r = circle/(physical_scale)/(pixel_scale)
            label = r'$r=' + str(round(circle)) + '\mathrm{\,kpc}$'
        else:
            r = circle/pixel_scale
            label = r'$r=' + str(round(circle)) + '\mathrm{\,arcsec}$'

        e = Ellipse(xy=(cen, cen), height=2*r, width=2*r)
        e.set_facecolor('none')
        e.set_edgecolor('w')
        e.set_label(label)
        ax1.add_patch(e)
        leg = ax1.legend(fontsize=20, frameon=False)
        leg.get_frame().set_facecolor('none')
        for text in leg.get_texts():
            text.set_color('w')

    if ax is not None:
        return ax

def SBP_shape(ell_free, ell_fix, redshift, pixel_scale, zeropoint, ax=None, x_min=1.0, x_max=4.0, physical_unit=False, show_dots=False, vertical_line=True, vertical_pos=100, linecolor='firebrick', label=None):
    """Display the 1-D profiles."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.48])
        ax2 = fig.add_axes([0.08, 0.55, 0.85, 0.20])
        ax3 = fig.add_axes([0.08, 0.75, 0.85, 0.20])
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        ax3.tick_params(direction='in')
    else:
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        ax3.tick_params(direction='in')

    import slug
    phys_size = slug.phys_size(redshift, is_print=False)
    # Calculate mean ellipticity and pa, which are used for fixed fitting
    interval = np.intersect1d(np.where(ell_free['sma'].data*pixel_scale*phys_size > 20),
               np.where(ell_free['sma'].data*pixel_scale*phys_size < 50))
    mean_e = ell_free['ell'][interval].mean()
    stdev_e = ell_free['ell'][interval].std()
    mean_pa = ell_free['pa_norm'][interval].mean()
    stdev_pa = ell_free['pa_norm'][interval].std()

    # 1-D profile
    if physical_unit is True:
        x = ell_fix['sma']*pixel_scale*phys_size
        y = -2.5*np.log10(ell_fix['intens'].data/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens']+ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens']-ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma']*pixel_scale
        y = -2.5*np.log10(ell_fix['intens']/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens']+ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens']-ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    
    # ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    
    if show_dots is True:
        ax1.errorbar((x ** 0.25), 
                 y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)
    ax1.plot(x**0.25, y, color=linecolor, linewidth=4, label=r'$\mathrm{'+label+'}$')
    ax1.fill_between(x**0.25, y_upper, y_lower, color=linecolor, alpha=0.3)
    ax1.axvline(x=vertical_pos**0.25, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], 
                    color='gray', linestyle='--', linewidth=3)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()
    if label is not None:
        ax1.legend(fontsize=20, frameon=False)
    
    # Ellipticity profile
    # ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
    if physical_unit is True:
        x = ell_free['sma']*pixel_scale*phys_size
    else:
        x = ell_free['sma']*pixel_scale
    if show_dots is True:
        ax2.errorbar((x ** 0.25), 
                     ell_free['ell'],
                     yerr=ell_free['ell_err'],
                     color='k', alpha=0.4, fmt='o', capsize=4, capthick=2, elinewidth=2)
    ax2.fill_between(x**0.25, ell_free['ell']+ell_free['ell_err'], ell_free['ell']-ell_free['ell_err'], 
                     color=linecolor, alpha=0.3)
    ax2.plot(x**0.25, ell_free['ell'], color=linecolor, linewidth=4)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0,0.7)
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i,2))+'$' for i in ytick_pos])
    # ax2.axhline(y = ell_free['ell'][~np.isnan(ell_free['ell'].data)].mean(),
    #           color=linecolor, alpha=1, linestyle = '-.', linewidth = 2)
    ax2.axhline(y = mean_e,
               color=linecolor, alpha=1, linestyle = '-.', linewidth = 2)

    # Position Angle profile
    #ax3.grid(linestyle='--', alpha=0.4, linewidth=2)
    from kungpao import utils
    pa_err = np.array([utils.normalize_angle(pa, lower=-90, 
                                             upper=90, b=True) for pa in ell_free['pa_err']])
    if show_dots is True:
        ax3.errorbar((x ** 0.25), 
                     ell_free['pa_norm'], yerr=pa_err,
                     color='k', alpha=0.4, fmt='o', capsize=4, capthick=2, elinewidth=2)
    ax3.fill_between(x**0.25, ell_free['pa_norm']+pa_err, ell_free['pa_norm']-pa_err, 
                     color=linecolor, alpha=0.3)
    ax3.plot(x**0.25, ell_free['pa_norm'], color=linecolor, linewidth=4)
    ax3.xaxis.set_major_formatter(NullFormatter())
    
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=25)
    
    #ax3.axhline(y = ell_free['pa_norm'][~np.isnan(ell_free['pa_norm'].data)].mean(),
    #           color=linecolor, alpha=1, linestyle = '-.', linewidth = 2)
    ax3.axhline(y = mean_pa,
               color=linecolor, alpha=1, linestyle = '-.', linewidth = 2)
    
    if physical_unit is True:
	    ax4 = ax3.twiny() 
	    ax4.tick_params(direction='in')
	    lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
	    lin_pos = [i**0.25 for i in lin_label]
	    ax4.set_xticks(lin_pos)
	    ax4.set_xlim(ax3.get_xlim())
	    ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
	    ax4.xaxis.set_label_coords(1, 1.05)
	    ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
	    for tick in ax4.xaxis.get_major_ticks():
	        tick.label.set_fontsize(25)
        
        
    if vertical_line is True:
        ax1.axvline(x=vertical_pos**0.25, ymin=0, ymax=1, 
                    color='gray', linestyle='--', linewidth=3)
        ax2.axvline(x=vertical_pos**0.25, ymin=0, ymax=1, 
                    color='gray', linestyle='--', linewidth=3)
        ax3.axvline(x=vertical_pos**0.25, ymin=0, ymax=1, 
                    color='gray', linestyle='--', linewidth=3)
        
    if ax is None:
        return fig
    return ax1, ax2, ax3



# You can plot 1-D SBP using this
def SBP_single(ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0, x_min=1.0, x_max=4.0, alpha=1, physical_unit=False, show_dots=False, vertical_line=False, vertical_pos=100, linecolor='firebrick', linestyle='-', label='SBP'):
    """Display the 1-D profiles."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.88])
        ax1.tick_params(direction='in')
    else:
        ax1 = ax
        ax1.tick_params(direction='in')

    # Calculate physical size at this redshift
    import slug
    phys_size = slug.phys_size(redshift,is_print=False)

    # 1-D profile
    if physical_unit is True:
        x = ell_fix['sma']*pixel_scale*phys_size
        y = -2.5*np.log10((ell_fix['intens'] + offset)/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens'] + offset + ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens'] + offset - ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma']*pixel_scale
        y = -2.5*np.log10((ell_fix['intens'] + offset)/(pixel_scale)**2)+zeropoint
        y_upper = -2.5*np.log10((ell_fix['intens'] + offset + ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        y_lower = -2.5*np.log10((ell_fix['intens'] + offset - ell_fix['int_err'])/(pixel_scale)**2)+zeropoint
        upper_yerr = y_lower-y
        lower_yerr = y-y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    
    # ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    
    if show_dots is True:
        ax1.errorbar((x ** 0.25), 
                 y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)
    if label is not None:
    	ax1.plot(x**0.25, y, color=linecolor, linewidth=4, linestyle=linestyle,
             label=r'$\mathrm{'+label+'}$', alpha=alpha)
    else:
    	ax1.plot(x**0.25, y, color=linecolor, linewidth=4, linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower, color=linecolor, alpha=0.3*alpha)
    ax1.axvline(x=vertical_pos**0.25, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], 
                    color='gray', linestyle='--', linewidth=3)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()
    if label is not None:
        ax1.legend(fontsize=25, frameon=False)
    
    if physical_unit is True:
        ax4 = ax1.twiny() 
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
        lin_pos = [i**0.25 for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
        ax4.xaxis.set_label_coords(1, 1.05)

        ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        
        
    if vertical_line is True:
        ax1.axvline(x=vertical_pos**0.25, ymin=0, ymax=1, 
                    color='gray', linestyle='--', linewidth=3)
        
    if ax is None:
        return fig
    return ax1

# Print attributes of a HDF5 file
def h5_print_attrs(f):
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.iteritems():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)


