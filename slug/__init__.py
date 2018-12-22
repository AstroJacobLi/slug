# Import packages
from __future__ import division, print_function
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


# Version
__version__ = "0.1"
__name__ = 'slug'

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


__all__ = ['phys_size','str2dic', 'skyobj_value', 'make_HSC_detect_mask', 
           'diagnose_image_mask','convert_HSC_binary_mask', 'print_HSC_binary_mask', 'gen_url_decals', 
           'login_naoj_server', 'gen_url_hsc_s18a', 'gen_url_hsc_s16a', 'h5_gen_mock_imag', 
            'rebin_img', 'extract_obj', 'make_binary_mask', 'evaluate_sky', 
           'evaluate_sky_dragonfly', 'run_SBP', 'display_isophote', 'SBP_shape', 'SBP_single',
           'h5_rewrite_dataset','h5_print_attrs']

########################################################################
########################## Basic Functions #############################
########################################################################
# Calculate physical size of a given redshift
def phys_size(redshift, is_print=True, H0=70, Omegam=0.3, Omegal=0.7):
    '''Calculate the corresponding physical size per arcsec of a given redshift
    in the Lambda-CDM cosmology.

    Requirement:
    -----------
    cosmology: https://github.com/esheldon/cosmology
    
    Parameters:
    -----------
    redshift: float
    is_print: boolean. If true, it will print out the physical scale at the given redshift.
    Omegam: float, density parameter of matter. It should be within [0, 1]. 
    Omegal: float, density parameter of Lambda.

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


########################################################################
########################## HD5 related #################################
########################################################################

# Print attributes of a HDF5 file
def h5_print_attrs(f):
    '''
    Print all attributes of a HDF5 file.

    Parameters:
    ----------
    f: HDF5 file.

    Returns:
    --------
    All attributes of 'f'
    '''
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.iteritems():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)

# Rewrite dataset
def h5_rewrite_dataset(mother_group, key, new_data):
    '''
    Rewrite the given dataset of a HDF5 group.

    Parameters:
    ----------
    mother_group: HDF5 group class.
    key: string, the name of the dataset to be writen into.
    new_data: The data to be written into.
    '''
    if np.any(np.array(mother_group.keys())==key):
        mother_group.__delitem__(key)
        mother_group.create_dataset(key, data=new_data)
    else:
        mother_group.create_dataset(key, data=new_data)

# String to dictionary
def str2dic(string):
    '''
    This function is used to load string dictionary and convert it into python dictionary.
    '''
    import yaml
    return yaml.load(string)

#########################################################################
########################## Mask related #################################
#########################################################################

# Make HSC detection and bright star mask
def make_HSC_detect_mask(bin_msk, img, objects, segmap, r=10.0, radius=1.5, threshold=0.01):
    '''Make HSC detection and bright star mask, 
    based on HSC binary mask flags.
    
    Parameters:
    -----------
    bin_msk: 2-D np.array, can be loaded from HSC image cutouts
    objects: table, returned from sep.extract_obj
    segmap: 2-D np.array, returned from sep.extract_obj
    r: float, blow-up parameter
    radius: float, convolution radius
    threshold: float, threshold of making mask after convolution

    Returns:
    -----------
    HSC_detect_mask: 2-D boolean np.array
    
    See also:
    -----------------
    convert_HSC_binary_mask(bin_msk)
    '''
    import sep
    TDmask = slug.convert_HSC_binary_mask(bin_msk)
    cen_mask = np.zeros(bin_msk.shape, dtype=np.bool)
    cen_obj = objects[segmap[int(bin_msk.shape[0] / 2.), int(bin_msk.shape[1] / 2.)] - 1]
    
    fraction_radius = sep.flux_radius(img, cen_obj['x'], cen_obj['y'], 10*cen_obj['a'], 0.5)[0]
    ba = np.divide(cen_obj['b'], cen_obj['a'])
    sep.mask_ellipse(cen_mask, cen_obj['x'], cen_obj['y'], fraction_radius, fraction_radius * ba,
                    cen_obj['theta'], r=r)
    from astropy.convolution import convolve, Gaussian2DKernel
    HSC_mask = (TDmask[:, :, 5]).astype(bool)*(~cen_mask) + TDmask[:, :, 9].astype(bool)
    # Convolve the image with a Gaussian kernel with the width of 1.5 pixel
    cvl = convolve(HSC_mask.astype('float'), Gaussian2DKernel(radius))
    HSC_detect_mask = cvl >= threshold
    return HSC_detect_mask

# Make HSC bright star mask
def make_HSC_bright_obj_mask(bin_msk, objects, segmap, r=10.0, radius=1.5, threshold=0.01):
    '''Make HSC bright star mask, based on HSC binary mask flags.
    
    Parameters:
    -----------
    bin_msk: 2-D np.array, can be loaded from HSC image cutouts
    objects: table, returned from sep.extract_obj
    segmap: 2-D np.array, returned from sep.extract_obj
    r: float, blow-up parameter
    radius: float, convolution radius
    threshold: float, threshold of making mask after convolution

    Returns:
    -----------
    HSC_detect_mask: 2-D boolean np.array
    
    See also:
    -----------------
    convert_HSC_binary_mask(bin_msk)
    '''
    TDmask = slug.convert_HSC_binary_mask(bin_msk)
    cen_mask = np.zeros(bin_msk.shape, dtype=np.bool)
    cen_obj = objects[segmap[int(bin_msk.shape[0] / 2.), int(bin_msk.shape[1] / 2.)] - 1]
    sep.mask_ellipse(cen_mask, cen_obj['x'], cen_obj['y'], cen_obj['a'], cen_obj['b'],
                    cen_obj['theta'], r=r)
    from astropy.convolution import convolve, Gaussian2DKernel
    HSC_mask = (TDmask[:, :, 9]).astype(bool)*(~cen_mask)
    # Convolve the image with a Gaussian kernel with the width of 1.5 pixel
    cvl = convolve(HSC_mask.astype('float'), Gaussian2DKernel(radius))
    HSC_detect_mask = cvl >= threshold
    return HSC_detect_mask

# Convert HSC binary mask to a 3-D array, with binary digits located in the third axis
def convert_HSC_binary_mask(bin_msk):
    '''Convert HSC binary mask to a 3-D array, 
    with binary digits located in the third axis.
    
    Parameters:
    -----------
    bin_msk: 2-D np.array, can be loaded from HSC image cutouts
    
    Returns:
    -----------
    TD: 3-D np.array, with binary digits located in the third axis.
    
    See also:
    -----------------
    print_HSC_binary_mask(TDmsk, path);
    slug.HSC_binray_mask_dict

    '''
    split_num = bin_msk.shape[1]
    a = np.array(bin_msk, dtype=np.uint16)
    b = np.array(np.hsplit(np.unpackbits(a.view(np.uint8), axis=1), split_num))
    TDim = np.flip(np.transpose(np.concatenate(np.flip(np.array(np.dsplit(b, 2)), axis=0), axis=2), axes=(1, 0, 2)), axis=2)
    return TDim

# Print HSC mask for each flag to 'png' files
def print_HSC_binary_mask(TDmsk, path):
    '''Print HSC mask for each flag.
    
    Parameters:
    -----------
    TDmsk: np.array, three dimensional (width, height, 16) mask array
    path: string, path of saving figures
    '''
    for i in range(16):
        _ = display_single(TDmsk[:,:,i].astype(float), cmap=SEG_CMAP, scale='linear')
        plt.savefig(path + 'HSC-bin-mask-' + HSC_binray_mask_dict[i] + '.png')

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

#########################################################################
########################## URL related #################################
#########################################################################

# Generate DECaLS tractor url, given brickname
def gen_url_decals_tractor(brickname):
    '''
    Generate DECaLS tractor url, given brickname. Work for python 2 and 3.
    '''
    return [
        'http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr7/tractor/'
        + brickname[:3] + '/tractor-' + brickname + '.fits'
    ]

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
    to enter username and password directly in Jupyter Notebook.
    In the configuration file, I wrote username in the first line, 
    and password in the second line.
    '''
    import urllib
    import urllib2
    # Import HSC username and password
    config = Table.read(config_path, format='ascii.no_header')['col1']
    username = config[0]
    password = config[1]
    # Create a password manager
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

    ###### For downloading images ######
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

    ###### For downloading PSFs ######
    # Add the username and password.
    top_level_url = 'https://hscdata.mtk.nao.ac.jp/psf/6/'
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
def gen_url_hsc_s18a(ra, dec, w, h, band, pixel_unit=False, only_image=False):
    '''Generate image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    w: float, width (arcsec)
    h: float, height (arcsec)
    band: string, such as 'r'
    pixel_unit: boolean, if your width and height are in pixel unit
    only_image: boolean, if you only want image layer

    Returns:
    -----------
    url: list of string, url of S18A image.  
    '''
    if only_image:
        if_variance_mask = 'off'
    else:
        if_variance_mask = 'on'

    if pixel_unit:
        return ['https://hscdata.mtk.nao.ac.jp/das_quarry/dr2.1/cgi-bin/cutout?ra='
            + str(ra) 
            + '&dec='
            + str(dec)
            + '&sw='
            + str(w*HSC_pixel_scale)
            + 'asec&sh='
            + str(h*HSC_pixel_scale)
            + 'asec&type=coadd&image=on&mask=' 
            + if_variance_mask + '&variance=' 
            + if_variance_mask + '&filter=HSC-'
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
           + 'asec&type=coadd&image=on&mask=' 
           + if_variance_mask + '&variance=' 
           + if_variance_mask + '&filter=HSC-'
           + str(band.upper())
           + '&tract=&rerun=s18a_wide']

# Generate HSC PSF url
def gen_psf_url_hsc_s18a(ra, dec, band):
    '''Generate PSF url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    band: string, such as 'r'
    
    Returns:
    -----------
    url: list of string, URL of S18A PSF.  
    '''

    return ['https://hscdata.mtk.nao.ac.jp/psf/6/cgi/getpsf?ra='
        + str(ra) 
        + '&dec='
        + str(dec)
        + '&filter='
        + str(band)
        + '&rerun=s18a_wide'
        + '&tract=&patch=&type=coadd']

# Generate HSC S16A image URL of given position.
def gen_url_hsc_s16a(ra, dec, w, h, band, pixel_unit=False):
    '''Generate HSC S16A image URL of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    w: float, width (arcsec)
    h: float, height (arcsec)
    band: string, such as 'r'

    Returns:
    -----------
    url: list of string, url of S16A image.  
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
            + 'asec&type=coadd&image=on&mask=on&variance=on&filter=HSC-'
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
           + 'asec&type=coadd&image=on&mask=on&variance=on&filter=HSC-'
           + str(band.upper())
           + '&tract=&rerun=s16a_wide2']


#########################################################################
########################## Mock Test related ############################
#########################################################################

# Generate mock images
def h5_gen_mock_image(h5_path, pixel_scale, band, i_gal_flux, i_gal_rh, 
    i_gal_q, i_sersic_index, i_gal_beta, i_psf_rh, groupname=None):
    '''
    Generate mock images.

    Parameters:
    -----------
    h5_path: string, the path of your h5 file.
    pixel_scale: float, in the unit of arcsec/pixel.
    band: string, such as 'r-band'.
    i_gal-flux: float, input galsim flux of the fake galaxy.
    i_gal_rh: float, input half-light-radius of the fake galaxy.
    i_gal_q: float, input b/a.
    i_sersic_index: float, input sersic index.
    i_gal_beta: float, input position angle (in degrees).
    i_psf_rh: float, the half-light-radius of PSF.
    groupname: string, such as 'model-0'.

    '''
    import h5py
    import galsim
    f = h5py.File(h5_path, 'r+')
    field = f['Background'][band]['image'][:]
    w = wcs.WCS(f['Background'][band]['image_header'].value)
    cen = field.shape[0] / 2  # Central position of the image
    print ('Size (in pixel):', [field.shape[0], field.shape[1]])
    print ('Angular size (in arcsec):', [
        field.shape[0] * pixel_scale, field.shape[1] * pixel_scale
    ])
    print ('The center of this image:', [field.shape[0] / 2, field.shape[1] / 2])
    # Define sersic galaxy
    gal = galsim.Sersic(i_sersic_index, half_light_radius=i_gal_rh, flux=i_gal_flux)
    # Shear the galaxy by some value.
    # q, beta      Axis ratio and position angle: q = b/a, 0 < q < 1
    gal_shape = galsim.Shear(q=i_gal_q, beta=i_gal_beta * galsim.degrees)
    gal = gal.shear(gal_shape)
    # Define the PSF profile
    #psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_rh)
    psf = galsim.Gaussian(sigma=i_psf_rh, flux=1.)
    # Convolve galaxy with PSF
    final = galsim.Convolve([gal, psf])
    # Draw the image with a particular pixel scale.
    image = final.drawImage(scale=pixel_scale, nx=field.shape[1], ny=field.shape[0])
    
    if groupname is None:
        groupname = 'n' + str(i_sersic_index)
    
    g1 = f['ModelImage'][band].create_group(groupname)
    g1.create_dataset('modelimage', data=image.array)

    # Generate mock image
    mock_img = image.array + field

    g2 = f['MockImage'][band].create_group(groupname)
    g2.create_dataset('mockimage', data=mock_img)

    # Plot fake galaxy and the composite mock image
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    display_single(image.array, ax=ax1, scale_bar_length=10)
    display_single(mock_img, scale_bar_length=10, ax=ax2)
    plt.show(block=False)
    plt.subplots_adjust(wspace=0.)
    f.close()


# Rebin a image / mask
def rebin_img(array, dimensions=None, scale=None):
    """ From http://martynbristow.co.uk/wordpress/blog/rebinning-data/
        It's a little bit slow, but flux is conserved.
        Return the array ``array`` to the new ``dimensions`` conserving 
        flux the flux in the bins.
        The sum of the array will remain the same.
        
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
        If the totals of the input and result array don't agree, raise an error 
        because computation may have gone wrong.
        
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


#########################################################################
########################## The Tractor related ##########################
#########################################################################

# Add sources to tractor
def add_tractor_sources(obj_cat, sources, w, shape_method='manual'):
    '''
    Add tractor sources to the sources list.

    Parameters:
    ----------
    obj_cat: astropy Table, objects catalogue.
    sources: list, to which we will add objects.
    w: wcs object.
    shape_method: string, 'manual' or 'decals'. If 'manual', it will adopt the 
                manually measured shapes. If 'decals', it will adopt 'DECaLS' 
                tractor shapes.

    Returns:
    --------
    sources: list of sources.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE
    obj_type = np.array(map(lambda st: st.rstrip(' '), obj_cat['type']))
    comp_galaxy = obj_cat[obj_type == 'COMP']
    dev_galaxy = obj_cat[obj_type == 'DEV']
    exp_galaxy = obj_cat[obj_type == 'EXP']
    rex_galaxy = obj_cat[obj_type == 'REX']
    psf_galaxy = obj_cat[np.logical_or(obj_type =='PSF', obj_type=='   ')]

    if shape_method is 'manual':
        # Using manually measured shapes
        if sources is None:
            sources = []
        for obj in comp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
                                90.0 + obj['theta'] * 180.0 / np.pi),
                    Flux(0.6 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
                                90.0 + obj['theta'] * 180.0 / np.pi)))
        for obj in dev_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in exp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in rex_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in psf_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print("Now you have %d sources" % len(sources))

    elif shape_method is 'decals':
        ## Using DECaLS shapes
        if sources is None:
            sources = []
        for obj in comp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             obj['shapeexp_e2']), Flux(0.6 * obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             obj['shapedev_e2'])))
        for obj in dev_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             -obj['shapedev_e2'])))
        for obj in exp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))
        for obj in rex_galaxy:
            #if obj['point_source'] > 0.0:
            #            sources.append(PointSource(PixPos(w.wcs_world2pix([[obj['ra'], obj['dec']]],1)[0]),
            #                                               Flux(obj['flux'])))
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))

        for obj in psf_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 1)[0]
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print("Now you have %d sources" % len(sources))
    else:
         raise ValueError('Cannot use this shape method') 
    return sources

# Do tractor iteration
def tractor_iteration(obj_cat, w, img_data, invvar, psf_obj, pixel_scale, kfold=4, fig_name=None):
    '''
    Run tractor iteratively.

    Parameters:
    -----------
    obj_cat: objects catalogue.
    w: wcs object.
    img_data: 2-D np.array, image.
    invvar: 2-D np.array, inverse variance matrix of the image.
    psf_obj: PSF object, defined by tractor.psf.PixelizedPSF() class.
    pixel_scale: float, pixel scale in unit arcsec/pixel.
    kfold: int, iteration time.
    fig_name: string, if not None, it will save the tractor subtracted image to the given path.

    Returns:
    -----------
    sources: list, containing tractor model sources.
    trac_obj: optimized tractor object after many iterations.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE

    step = int((len(obj_cat) - 50)/(kfold-1))
    for i in range(kfold):
        if i == 0:
            obj_small_cat = obj_cat[:50]
            sources = slug.tractor.add_tractor_sources(obj_small_cat, None, w, shape_method='manual')
        else:
            obj_small_cat = obj_cat[50 + step*(i-1) : 50 + step*(i)]
            sources = slug.tractor.add_tractor_sources(obj_small_cat, sources, w, shape_method='manual')

        tim = Image(data=img_data,
                    invvar=invvar,
                    psf=psf_obj,
                    wcs=NullWCS(pixscale=pixel_scale),
                    sky=ConstantSky(0.0),
                    photocal=NullPhotoCal()
                    )
        trac_obj = Tractor([tim], sources)
        trac_mod = trac_obj.getModelImage(0, minsb=0.0)

        # Optimization
        trac_obj.freezeParam('images')
        trac_obj.optimize_loop()
        ########################
        plt.clf()
        plt.rc('font', size=20)
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18,8))

        trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[:])

        ax1 = display_single(img_data, ax=ax1, scale_bar=False)
        ax1.set_title('raw image')
        ax2 = display_single(trac_mod_opt, ax=ax2, scale_bar=False, contrast=0.02)
        ax2.set_title('tractor model')
        ax3 = display_single(abs(img_data - trac_mod_opt), ax=ax3, scale_bar=False, color_bar=True, contrast=0.05)
        ax3.set_title('residual')
        #trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[1:])
        #ax4 = display_single(img_data - trac_mod_opt, ax=ax4, scale_bar=False, color_bar=True, contrast=0.05)
        #ax4.set_title('remain central galaxy')
        if i == (kfold-1):
            if fig_name is not None:
                plt.savefig('./Figures/' + fig_name, dpi=200, bbox_inches='tight')
        plt.show(block=False)
        print('The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))))

    return sources, trac_obj

#########################################################################
########################## 1-D profile related ##########################
#########################################################################

# Extract objects for a given image
def extract_obj(img, b=30, f=5, sigma=5, show_fig=True, pixel_scale=0.168, minarea=5, 
    deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0):
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



# Evaluate the mean sky value
def evaluate_sky(img, sigma=1.5, radius=15, threshold=0.005, clean_param=1.0, show_fig=True, show_hist=True):
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

# Calculate mean/median value of nearby sky objects
def skyobj_value(sky_cat, cen_ra, cen_dec, matching_radius=3, aperture='84', 
    print_number=False, sigma_upper=3., sigma_lower=3., iters=5, showmedian=False):
    '''Calculate the mean/median value of nearby SKY OBJECTS around a given RA and DEC.
    Importing sky objects catalog can be really slow.

    Parameters:
    -----------
    path: string, the path of catalog.
    cen_ra, cen_dec: float, RA and DEC of the given object.
    matching_radius: float, in arcmin. We match sky objects around the given object within this radius.
    aperture: string, must be in the `SkyObj_aperture_dic`.
    print_number: boolean. If true, it will print the number of nearby sky objects.
    sigma_upper, sigma_lower: float, threshold for sigma_clipping of nearby sky objects.
    iters: positive int, time of iterations.
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
    obj_inx = np.where(catalog.separation(bkg_pos) < matching_radius * u.arcmin)[0]
    if print_number:
        print('Sky objects number around' + str(matching_radius) + 'arcmin: ', len(obj_inx))


    x = sky_cat[obj_inx]['r_apertureflux_' + aperture +'_flux'] * 1.7378e30 / (np.pi * SkyObj_aperture_dic[aperture]**2)
    x = sigma_clip(x, sigma_lower=sigma_lower, sigma_upper=sigma_upper, iters=iters)
    if showmedian:
        return np.median(x)
    else:
        return np.mean(x)

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

# Run surface brightness profile for the given image and mask
def run_SBP(img_path, msk_path, pixel_scale, phys_size, iraf_path, step=0.10, 
    sma_ini=10.0, sma_max=900.0, n_clip=3, low_clip=3.0, upp_clip=2.5, force_e=None, r_interval=(20, 50), outPre=None):
    # Centeral coordinate 
    img_data = fits.open(img_path)[0].data
    x_cen, y_cen = int(img_data.shape[1]/2), int(img_data.shape[0]/2)

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

#########################################################################
############################ Display related ###########################
#########################################################################

def display_isophote(img, ell, pixel_scale, scale_bar=True, scale_bar_length=50, 
    physical_scale=None, text=None, ax=None, contrast=None, circle=None):
    """
    Visualize the isophotes.
    
    Parameters:
    ----------
    img: 2-D np.array, image.
    ell: astropy Table or numpy table, is the output of ELLIPSE.
    pixel_scale: float, pixel scale in arcsec/pixel.
    scale_bar: boolean, whether show scale bar.
    scale_bar_length: float, length of scale bar.
    physical_scale: float. If not None, the scale bar will be shown in physical scale.
    text: string. If not None, the string will be shown in the upper left corner.
    contrast: float. Default contrast is 0.15.
    circle: list of floats. Maximun length is 3.

    Returns:
    --------
    ax: matplotlib axes class.

    """
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax

    ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.xaxis.set_major_formatter(NullFormatter())

    cen = int(img.shape[0]/2),

    if contrast is not None:
        ax1 = display_single(img, pixel_scale=pixel_scale, ax=ax1, scale_bar=scale_bar, 
            scale_bar_length=scale_bar_length, physical_scale=physical_scale, 
            contrast=contrast, add_text=text)
    else:
        ax1 = display_single(img, pixel_scale=pixel_scale, ax=ax1, scale_bar=scale_bar, 
            scale_bar_length=scale_bar_length, physical_scale=physical_scale, 
            contrast=0.15, add_text=text)
    
    for k, iso in enumerate(ell):
        if k % 2 == 0:
            e = Ellipse(xy=(iso['x0'], iso['y0']),
                        height=iso['sma'] * 2.0,
                        width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                        angle=iso['pa'])
            e.set_facecolor('none')
            e.set_edgecolor('r')
            e.set_alpha(0.4)
            e.set_linewidth(1.1)
            ax1.add_artist(e)
    ax1.set_aspect('equal')

    if circle is not None:
        if physical_scale is not None:
            r = np.array(circle) / (physical_scale) / (pixel_scale)
            label_suffix = r'\mathrm{\,kpc}$'
        else:
            r = np.array(circle) / pixel_scale
            label_suffix = r'\mathrm{\,arcsec}$'

        style_list = ['-', '--', '-.']

        for num, rs in enumerate(r):
            e = Ellipse(xy=(img.shape[1]/2, img.shape[0]/2), 
                        height=2*rs, width=2*rs, 
                        linestyle=style_list[num], linewidth=1.5)
            label = r'$r=' + str(round(circle[num])) + label_suffix
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

# You can plot 1-D SBP using this, without plotting the PA and eccentricity.
def SBP_single(ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0, 
    x_min=1.0, x_max=4.0, alpha=1, physical_unit=False, show_dots=False, show_grid=False, 
    show_banner=True, vertical_line=None, linecolor='firebrick', linestyle='-', 
    linewidth=3, label='SBP'):

    """Display the 1-D profiles, without showing PA and ellipticity.
    
    Parameters:
    -----------
    ell_fix: astropy Table or numpy table, should be the output of ELLIPSE.
    redshift: float, redshift of the object.
    pixel_scale: float, pixel scale in arcsec/pixel.
    zeropoint: float, zeropoint of the photometry system.
    ax: matplotlib axes class.
    offset: float.
    x_min, x_max: float, in ^{1/4} scale.
    alpha: float, transparency.
    physical_unit: boolean. If true, the figure will be shown in physical scale.
    show_dots: boolean. If true, it will show all the data points.
    show_grid: boolean. If true, it will show a grid.
    vertical_line: list of floats, positions of vertical lines. Maximum length is three.
    linecolor, linestyle: string. Color and style of SBP.
    label: string.

    Returns:
    --------
    ax: matplotlib axes class.

    """
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
    phys_size = slug.phys_size(redshift,is_print=False)

    # 1-D profile
    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * np.log10((ell_fix['intens'] + offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * np.log10((ell_fix['intens'] + offset + ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * np.log10((ell_fix['intens'] + offset - ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'] + offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * np.log10((ell_fix['intens'] + offset + ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * np.log10((ell_fix['intens'] + offset - ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    if show_grid:
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    if show_dots:
        ax1.errorbar((x ** 0.25), y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)

    if label is not None:
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth, linestyle=linestyle,
             label=r'$\mathrm{' + label + '}$', alpha=alpha)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower, color=linecolor, alpha=0.3*alpha)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()

    # Twin axis with linear scale
    if physical_unit and show_banner is True:
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

    # Vertical line
    if vertical_line is not None:
        if len(vertical_line) > 3:
            raise ValueError('Maximum length of vertical_line is 3.') 
        ylim = ax1.get_ylim()
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos**0.25, ymin=0, ymax=1,
                        color='gray', linestyle=style_list[k], linewidth=3, alpha=0.75)
        plt.ylim(ylim)

    # Return
    if ax is None:
        return fig
    return ax1


# You can plot 1-D SBP using this, containing SBP, PA and eccentricity.
def SBP_shape(ell_free, ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0,
    x_min=1.0, x_max=4.0, alpha=1.0, r_interval=(20, 50), physical_unit=False, 
    show_dots=False, show_grid=False, show_hline=True, vertical_line=None, linecolor='firebrick', linestyle='-', 
    linewidth=3, label=None):
    """
    Display the 1-D profiles, containing SBP, PA and eccentricity.
    
    Parameters:
    -----------
    ell_free, ell_fix: astropy Table or numpy table, should be the output of ELLIPSE. 
                        'Free' indicates free shapes during fitting, 'fix' indicates fixed shapes.
    redshift: float, redshift of the object.
    pixel_scale: float, pixel scale in arcsec/pixel.
    zeropoint: float, zeropoint of the photometry system.
    ax: matplotlib axes class.
    offset: float.
    x_min, x_max: float, in ^{1/4} scale.
    alpha: float, transparency.
    r_interval: number tuple, within with mean PA and e are calculated. I suggest setting it to the same value as fitting.
    physical_unit: boolean. If true, the figure will be shown in physical scale.
    show_dots: boolean. If true, it will show all the data points.
    show_grid: boolean. If true, it will show a grid.
    show_hline: boolean. If true, it will show the mean PA and e within `r_interval`.
    vertical_line: list of floats, positions of vertical lines. Maximum length is three.
    linecolor, linestyle: string. Color and style of SBP.
    label: string.

    Returns:
    --------
    ax: matplotlib axes class.
    """

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

    # Calculate physical size
    phys_size = slug.phys_size(redshift, is_print=False)
    # Calculate mean ellipticity and pa, which are used for fixed fitting
    interval = np.intersect1d(np.where(ell_free['sma']*pixel_scale*phys_size > r_interval[0]),
               np.where(ell_free['sma']*pixel_scale*phys_size < r_interval[1]))
    mean_e = ell_free['ell'][interval].mean()
    stdev_e = ell_free['ell'][interval].std()
    mean_pa = ell_free['pa_norm'][interval].mean()
    stdev_pa = ell_free['pa_norm'][interval].std()
    
    # 1-D profile
    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * np.log10((ell_fix['intens'] + offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * np.log10((ell_fix['intens'] + offset + ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * np.log10((ell_fix['intens'] + offset - ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'] + offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * np.log10((ell_fix['intens'] + offset + ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * np.log10((ell_fix['intens'] + offset - ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    if show_dots:
        ax1.errorbar((x ** 0.25), y,
                 yerr=asymmetric_error,
                 color='k', alpha=0.2, fmt='o', 
                 capsize=4, capthick=1, elinewidth=1)
    if label is not None:
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth, linestyle=linestyle,
             label=r'$\mathrm{' + label + '}$', alpha=alpha)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        
    ax1.fill_between(x**0.25, y_upper, y_lower, color=linecolor, alpha=alpha*0.3)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()

    # Ellipticity profile
    if physical_unit is True:
        x = ell_free['sma'] * pixel_scale * phys_size
    else:
        x = ell_free['sma'] * pixel_scale
    if show_dots is True:
        ax2.errorbar((x ** 0.25), 
                     ell_free['ell'],
                     yerr=ell_free['ell_err'],
                     color='k', alpha=0.4, fmt='o', capsize=4, capthick=2, elinewidth=2)
    ax2.fill_between(x**0.25, ell_free['ell'] + ell_free['ell_err'], 
                        ell_free['ell'] - ell_free['ell_err'], 
                        color=linecolor, alpha=alpha*0.3)
    ax2.plot(x**0.25, ell_free['ell'], color=linecolor, linewidth=linewidth, alpha=alpha)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ylim = ax2.get_ylim()
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0, 0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i,2))+'$' for i in ytick_pos])
    ax2.set_ylim(ylim)
    if show_hline:
        ax2.axhline(y = mean_e, color=linecolor, 
            alpha=1, linestyle = '-.', linewidth = 2)

    # Position Angle profile
    from kungpao import utils
    pa_err = np.array([utils.normalize_angle(pa, lower=-90, 
                                             upper=90, b=True) for pa in ell_free['pa_err']])
    if show_dots is True:
        ax3.errorbar((x ** 0.25), 
                     ell_free['pa_norm'], yerr=pa_err,
                     color='k', alpha=0.4, fmt='o', capsize=4, capthick=2, elinewidth=2)
    ax3.fill_between(x**0.25, ell_free['pa_norm'] + pa_err, ell_free['pa_norm'] - pa_err,
                     color=linecolor, alpha=0.3*alpha)
    ax3.plot(x**0.25, ell_free['pa_norm'], color=linecolor, linewidth=linewidth, alpha=alpha)
    ax3.xaxis.set_major_formatter(NullFormatter())
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax3.set_xlim(x_min, x_max)
    ylim = ax3.get_ylim()
    ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=25)
    ytick_pos = [-90, -60, -30, 0, 30, 60, 90]
    ax3.set_yticks(ytick_pos)
    ax3.set_ylim(ylim)
    if show_hline:
        ax3.axhline(y = mean_pa, color=linecolor, 
            alpha=1, linestyle = '-.', linewidth = 2)
        
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
        
    if vertical_line:
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            ax2.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            ax3.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
    if show_grid:
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
        ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
        ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ax is None:
        return fig
    return ax1, ax2, ax3


# Plot SBP together, and also plot median profile
def SBP_stack(obj_cat, band, filenames, pixel_scale, zeropoint, ax=None, physical_unit=False, 
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, show_single=True, 
    vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', linewidth=5,
    single_alpha=0.3, single_color='firebrick', single_style='-', single_width=1, label=None):
    """
    Plot SBP together, along with median profile
    
    Parameters:
    -----------
    obj_cat: object catalog.
    band: string, such as 'r-band'.
    filenames: list containing corresponding filenames in the obj_cat.
    pixel_scale: float, pixel scale in arcsec/pixel.
    zeropoint: float, zeropoint of the photometry system.
    ax: matplotlib axes class.
    physical_unit: boolean. If true, the figure will be shown in physical scale.
    sky_cat: SkyObject catalog.
    matching_radius: float, in arcmin. We match sky objects around the given object within this radius.
    aperture: string, must be in the `SkyObj_aperture_dic`.
    x_min, x_max: float, in ^{1/4} scale.
    show_single: boolean. If yes, it will show every single profile.
    vertical_line: list of positions. Maximum length is three.
    linecolor, fillcolor, linewidth: arguments for the median profile.
    sing_alpha, single_color, single_style, single_width: arguments for single profiles.
    label: string.

    Returns:
    --------
    ax: matplotlib axes class.
    y_stack: stacked profile ndarray.
    """
    import h5py
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        fig.subplots_adjust(left=0.0, right=1.0, 
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.88])
        ax1.tick_params(direction='in')
    else:
        ax1 = ax
        ax1.tick_params(direction='in')


    for k, obj in enumerate(obj_cat):
        # Load files
        f = h5py.File(filenames[k], 'r')
        # Load info
        info = slug.str2dic(f['info'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        img = f['Image'][band]['image'].value
        mask = f['Mask'][band].value
        ell_free = f['ell_free'][band].value
        ell_fix = f['ell_fix'][band].value
        if sky_cat is None:
            off_set = 0.0
        else:
            off_set = slug.skyobj_value(sky_cat,
                                        ra,
                                        dec,
                                        matching_radius=matching_radius,
                                        aperture=aperture,
                                        iters=5,
                                        showmedian=False)
        if k == 0:
            label = "S18A\ sky\ objects"
        else:
            label = None
        if show_single:
            slug.SBP_single(
                ell_fix,
                redshift,
                pixel_scale,
                zeropoint,
                ax=ax1,
                offset=-off_set,
                physical_unit=physical_unit,
                x_min=x_min,
                x_max=x_max,
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=label)

        x = ell_fix['sma'] * pixel_scale * slug.phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens'], kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, 60)
        if k == 0:
            y_stack = func(x_input)
        else:
            y_stack = np.vstack((y_stack, func(x_input)))
        f.close()

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        yerr_set = np.array([np.std(bootstrap(bootarr, 100, bootfunc=np.nonmedian)) for bootarr in y_stack.T])

    y = -2.5*np.log10(np.median(y_stack, axis=0)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    y_upper = -2.5*np.log10((np.median(y_stack, axis=0) + yerr_set)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    y_lower = -2.5*np.log10((np.median(y_stack, axis=0) - yerr_set)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    upper_yerr = y_lower - y
    lower_yerr = y - y_upper
    asymmetric_error = [lower_yerr, upper_yerr]
    
    if label is not None:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-',
             label=r'$\mathrm{' + label + '}$', alpha=1)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-', alpha=1)
    ax1.fill_between(x_input, y_upper, y_lower, color=fillcolor, alpha=0.4)

    # Return
    if ax is None:
        return fig, y_stack
    return ax1, y_stack


# You can plot 1-D SBP using this, containing SBP, PA and eccentricity.
def SBP_stack_shape(obj_cat, band, filenames, pixel_scale, zeropoint, ax=None, physical_unit=False, 
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, ninterp=30,
    show_single=True, vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', 
    linewidth=5, single_alpha=0.3, single_color='firebrick', single_style='-', single_width=1, label=None):
    """
    Plot profiles of surface brightness, PA and ellipticity together, 
    along with median profiles.
    
    Parameters:
    -----------
    obj_cat: object catalog.
    band: string, such as 'r-band'.
    filenames: list containing corresponding filenames in the obj_cat.
    pixel_scale: float, pixel scale in arcsec/pixel.
    zeropoint: float, zeropoint of the photometry system.
    ax: list. It should be a list containing three matplotlib axes.
    physical_unit: boolean. If true, the figure will be shown in physical scale.
    sky_cat: SkyObject catalog.
    matching_radius: float, in arcmin. We match sky objects around the given object within this radius.
    aperture: string, must be in the `SkyObj_aperture_dic`.
    x_min, x_max: float, in ^{1/4} scale.
    show_single: boolean. If yes, it will show every single profile.
    vertical_line: list of positions. Maximum length is three.
    linecolor, fillcolor, linewidth: arguments for the median profile.
    sing_alpha, single_color, single_style, single_width: arguments for single profiles.
    label: string.

    Returns:
    --------
    ax: matplotlib axes class.
    y_stack: stacked profile ndarray.
    """

    
    import h5py
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext

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


    for k, obj in enumerate(obj_cat):
        # Load files
        f = h5py.File(filenames[k], 'r')
        # Load info
        info = slug.str2dic(f['info'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        img = f['Image'][band]['image'].value
        mask = f['Mask'][band].value
        ell_free = f['ell_free'][band].value
        ell_fix = f['ell_fix'][band].value

        # Calculate mean ellipticity and pa, which are used for fixed fitting
        mean_e = info['mean_e']
        mean_pa = info['mean_pa']
        
        if sky_cat is None:
            off_set = 0.0
        else:
            off_set = slug.skyobj_value(sky_cat,
                                        ra,
                                        dec,
                                        matching_radius=matching_radius,
                                        aperture=aperture,
                                        iters=5,
                                        showmedian=False)
        if k == 0:
            single_label = None #"S18A\ sky\ objects"
        else:
            single_label = None

        if physical_unit is False:
            raise ValueError('You must use physical sacle.')

        if show_single:
            slug.SBP_shape(
                ell_free,
                ell_fix,
                redshift,
                pixel_scale,
                zeropoint,
                ax=[ax1, ax2, ax3],
                physical_unit=physical_unit,
                x_min=x_min,
                x_max=x_max,
                show_hline=False,
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)
        
        x_input = np.linspace(x_min, x_max, ninterp)

        # Interpolate for surface brightness
        x = ell_fix['sma'] * pixel_scale * slug.phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens'], kind='cubic', fill_value='extrapolate')
        if k == 0:
            SB_stack = func(x_input)
        else:
            SB_stack = np.vstack((SB_stack, func(x_input)))

        # Interpolate for ellipticity
        x = ell_free['sma'] * slug.HSC_pixel_scale * slug.phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['ell'])
        func = interpolate.interp1d(x[mask]**0.25, ell_free['ell'][mask], kind='cubic', fill_value='extrapolate')
        if k == 0:
            e_stack = func(x_input)
        else:
            e_stack = np.vstack((e_stack, func(x_input)))
            
        # Interpolate for surface brightness
        x = ell_free['sma'] * slug.HSC_pixel_scale * slug.phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['pa_norm'])
        func = interpolate.interp1d(x[mask]**0.25, ell_free['pa_norm'][mask], kind='cubic', fill_value='extrapolate')
        if k == 0:
            pa_stack = func(x_input)
        else:
            pa_stack = np.vstack((pa_stack, func(x_input)))
        
        f.close()

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.mean
        SB_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in SB_stack.T])
        e_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in e_stack.T])
        pa_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in pa_stack.T])
    
    # ax1: SBP
    y = -2.5*np.log10(np.nanmedian(SB_stack, axis=0)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    y_upper = -2.5*np.log10((np.nanmedian(SB_stack, axis=0) + SB_err)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    y_lower = -2.5*np.log10((np.nanmedian(SB_stack, axis=0) - SB_err)/(slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    upper_yerr = y_lower - y
    lower_yerr = y - y_upper
    asymmetric_error = [lower_yerr, upper_yerr]
    
    if label is not None:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-',
             label=r'$\mathrm{' + label + '}$', alpha=1)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-', alpha=1)
    ax1.fill_between(x_input, y_upper, y_lower, color=fillcolor, alpha=0.4)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
    ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel(ylabel, fontsize=30)
    ax1.invert_yaxis()
    
    # ax2: ellipticity
    y = np.nanmedian(e_stack, axis=0)
    ax2.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-', alpha=1)
    ax2.fill_between(x_input, y - e_err, y + e_err, color=fillcolor, alpha=0.4)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ylim = ax2.get_ylim()
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0, 0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i,2))+'$' for i in ytick_pos])
    ax2.set_ylim(ylim)
    #ax2.axhline(y = mean_e, color=linecolor, 
    #    alpha=1, linestyle = '-.', linewidth = 2)

    # ax3: Position angle
    y = np.nanmedian(pa_stack, axis=0)
    ax3.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-', alpha=1)
    ax3.fill_between(x_input, y - pa_err, y + pa_err, color=fillcolor, alpha=0.4)
    ax3.xaxis.set_major_formatter(NullFormatter())
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax3.set_xlim(x_min, x_max)
    ylim = ax3.get_ylim()
    ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=25)
    ytick_pos = [-90, -60, -30, 0, 30, 60, 90]
    ax3.set_yticks(ytick_pos)
    ax3.set_ylim(ylim)
    #ax3.axhline(y = mean_pa, color=linecolor, 
    #    alpha=1, linestyle = '-.', linewidth = 2)

    # Return
    if ax is None:
        return fig, SB_stack, e_stack, pa_stack
    return [ax1, ax2, ax3], SB_stack, e_stack, pa_stack