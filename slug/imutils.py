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
from astropy.table import Table, Column, hstack, vstack

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

import sep

from kungpao import imtools
from kungpao import io
#from .display import display_single, SEG_CMAP

from .__init__ import SkyObj_aperture_dic

__all__ = ["phys_size", "convert_HSC_binary_mask", "make_HSC_detect_mask", 
            "make_HSC_bright_obj_mask", "print_HSC_binary_mask", "make_binary_mask",
            "gen_url_decals_tractor", "gen_url_decals_jpeg", "gen_url_decals",
            "login_naoj_server_3", "login_naoj_server", 
            "gen_url_hsc_s18a", "gen_url_hsc_s16a", "gen_psf_url_hsc_s18a"]

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

def img_cutout(img, wcs, coord_1, coord_2, size=60.0, pix=0.168,
               prefix='img_cutout', pixel_unit=False, img_header=None, 
               out_dir=None, save=True):
    """(From kungpao) Generate image cutout with updated WCS information.

    ----------
    Parameters:
        pixel_unit: boolen, optional
                    When True, coord_1, cooord_2 becomes X, Y pixel coordinates.
                    Size will also be treated as in pixels.
        img: 2d array.
        wcs: astropy wcs object of the input image.
        coord_1: ra of the center.
        coord_2: dec of the center.
        size: image size.
        pix: pixel size.
        img_header: the astropy header object of the input image. 
                    In case you can save the infomation in this header to the new header.
    """
    from astropy.nddata import Cutout2D
    if not pixel_unit:
        # imgsize in unit of arcsec
        cutout_size = np.asarray(size) / pix
        cen_x, cen_y = wcs.wcs_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(keyword, img_header[keyword], img_header.comments[keyword])
    
    # Build a HDU
    hdu = fits.PrimaryHDU(header=cutout_header)
    hdu.data = cutout.data

    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout

# Calculate mean/median value of nearby sky objects
def skyobj_value(sky_cat, cen_ra, cen_dec, matching_radius=[1, 3], aperture='84', redshift=None,
    print_number=False, sigma_upper=3., sigma_lower=3., maxiters=5, showmedian=False, verbose=False):
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
    if redshift is not None:
        matching_radius = np.asarray(matching_radius)
        matching_radius = matching_radius / phys_size(redshift, is_print=False)
        matching_radius = matching_radius / 60
        matching_radius = list(matching_radius)
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
    if verbose:
        print('{} sky objects got matched!'.format(len(x)))
    if showmedian:
        return np.nanmedian(x)
    else:
        return np.nanmean(x)

# Calculate mean/median value of nearby sky objects
def skyobj_std(sky_cat, cen_ra, cen_dec, matching_radius=[1, 3], aperture='84', 
    print_number=False, sigma_upper=3., sigma_lower=3., maxiters=5, showmedian=False, verbose=False):
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
    if verbose:
        print('{} sky objects got matched!'.format(len(x)))
    return np.std(x)


#########################################################################
########################## Mask related #################################
#########################################################################
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
    HSC_binray_mask_dict

    '''
    split_num = bin_msk.shape[1]
    a = np.array(bin_msk, dtype=np.uint16)
    b = np.array(np.hsplit(np.unpackbits(a.view(np.uint8), axis=1), split_num))
    TDim = np.flip(np.transpose(np.concatenate(np.flip(np.array(np.dsplit(b, 2)), axis=0), axis=2), axes=(1, 0, 2)), axis=2)
    return TDim

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
    TDmask = convert_HSC_binary_mask(bin_msk)
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
    TDmask = convert_HSC_binary_mask(bin_msk)
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

# Print HSC mask for each flag to 'png' files
def print_HSC_binary_mask(TDmsk, path):
    '''Print HSC mask for each flag.
    
    Parameters:
    -----------
    TDmsk: np.array, three dimensional (width, height, 16) mask array
    path: string, path of saving figures
    '''
    from .display import display_single, IMG_CMAP, SEG_CMAP
    for i in range(16):
        _ = display_single(TDmsk[:,:,i].astype(float), cmap=SEG_CMAP, scale='linear')
        plt.savefig(path + 'HSC-bin-mask-' + HSC_binray_mask_dict[i] + '.png')

# Make binary mask
def make_binary_mask(img, w, segmap, radius=10.0, threshold=0.01, 
    gaia=True, factor_b=1.2, sep_objcat=None, sep_mag=18.0, sep_zp=27.0, sep_blowup=15, 
    show_fig=True):
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

    if sep_objcat is not None:
        t = Table(sep_objcat)
        cen_inx = segmap[int(img.shape[0] / 2.), int(img.shape[1] / 2.)] - 1
        cen_obj = sep_objcat[cen_inx]
        t.remove_row(cen_inx)
        t.sort('flux')
        t.reverse()
        bright_objs = t[sep_zp - 2.5 * np.log10(t['flux']) < sep_mag]
        print('The number of bright objects: ', len(bright_objs))
        for skyobj in bright_objs:
            sep.mask_ellipse(seg_mask, skyobj['x'], skyobj['y'], 
                             skyobj['a'], skyobj['b'], skyobj['theta'], 
                             r=sep_blowup)
    if gaia is False:
        if show_fig:
            from .display import display_single, IMG_CMAP, SEG_CMAP
            display_single(seg_mask.astype(int), cmap=SEG_CMAP)
        return seg_mask
    else:
        # Combine this mask with Gaia star mask
        gaia_mask = imtools.gaia_star_mask(img, w, gaia_bright=16, factor_f=10000, factor_b=factor_b)[1].astype('bool')
        if show_fig:
            from .display import display_single, IMG_CMAP, SEG_CMAP
            display_single((seg_mask + gaia_mask).astype(int), cmap=SEG_CMAP)

        binary_mask = seg_mask + gaia_mask
        return binary_mask

# evaluate_sky objects for a given image
def extract_obj(img, b=30, f=5, sigma=5, pixel_scale=0.168, minarea=5, 
    deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0, 
    sky_subtract=False, show_fig=True, verbose=True, flux_auto=True, flux_aper=None):
    '''Extract objects for a given image, using `sep`. This is from `slug`.

    Parameters:
    ----------
    img: 2-D numpy array
    b: float, size of box
    f: float, size of convolving kernel
    sigma: float, detection threshold
    pixel_scale: float

    Returns:
    -------
    objects: astropy Table, containing the positions,
        shapes and other properties of extracted objects.
    segmap: 2-D numpy array, segmentation map
    '''

    # Subtract a mean sky value to achieve better object detection
    b = 30  # Box size
    f = 5   # Filter width
    bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    data_sub = img - bkg.back()
    
    sigma = sigma
    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img
    objects, segmap = sep.extract(input_data,
                                  sigma,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea)
    if verbose:                              
        print("# Detect %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)) + 1, name='index'))
    # Maximum flux, defined as flux within six 'a' in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], 
                                    6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'. 
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2* np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)), 
                              name='fwhm_custom'))
    
    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'], objects['y'], 
                                          objects['a'], objects['b'], 
                                          objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data, objects['x'], objects['y'], 
                                            2.5 * (kronrad), subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data, objects['x'][use_circle], objects['y'][use_circle],
                                                r_min, subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))
        
    if flux_aper is not None:
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0], flux_aper[1])[0], name='flux_ann'))
        '''
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0] * objects['a'])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1] * objects['a'])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0] * objects['a'], flux_aper[1] * objects['a'])[0], name='flux_ann'))
        '''

    # plot background-subtracted image
    if show_fig:
        from .display import display_single, IMG_CMAP, SEG_CMAP
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0] = display_single(data_sub, ax=ax[0], scale_bar=False, pixel_scale=pixel_scale)
        from matplotlib.patches import Ellipse
        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=7*obj['a'],
                        height=7*obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP , ax=ax[1])
    return objects, segmap


#########################################################################
########################## URL related #################################
#########################################################################
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

# Generate DECaLS tractor url, given brickname
def gen_url_decals_tractor(brickname):
    '''
    Generate DECaLS tractor url, given brickname. Work for python 2 and 3.
    '''
    return [
        'http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr7/tractor/'
        + brickname[:3] + '/tractor-' + brickname + '.fits'
    ]

# Generate DECaLS jpeg cutout url
def gen_url_decals_jpeg(ra_cen, dec_cen, size, bands, layer='decals-dr7', pixel_unit=False):
    '''Generate jpeg image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    size: float, image size (pixel)
    bands: string, such as 'r' or 'gri'
    layer: string, edition of data release
    pixel_unit: boolean. If true, size will be in pixel unit.

    Returns:
    -----------
    url: list of str, url of S18A image.  
    '''
    if pixel_unit:
        return ['http://legacysurvey.org/viewer/jpeg-cutout?ra=' 
                + str(ra_cen) 
                + '&dec=' 
                + str(dec_cen) 
                + '&layer=' 
                + layer 
                + '&size=' 
                + str(size) 
                + '&pixscale=' 
                + str(0.262) 
                + '&bands=' 
                + bands]
    else:
        return ['http://legacysurvey.org/viewer/jpeg-cutout?ra=' 
                + str(ra_cen) 
                + '&dec=' 
                + str(dec_cen) 
                + '&layer=' 
                + layer 
                + '&size=' 
                + str(int(size/0.262))
                + '&pixscale=' 
                + str(0.262) 
                + '&bands=' 
                + bands]

# Generate DECaLS image url
def gen_url_decals(ra, dec, size, bands, layer='decals-dr7', pixel_unit=False):
    '''Generate fits image url of given position.
    
    Parameters:
    -----------
    ra: float, RA (degrees)
    dec: float, DEC (degrees)
    size: float, image size (pixel)
    bands: string, such as 'r' or 'gri'
    layer: string, edition of data release
    pixel_unit: boolean. If true, size will be in pixel unit.

    Returns:
    -----------
    url: list of str, url of DECaLS image.  
    '''

    if pixel_unit:
        return ['http://legacysurvey.org/viewer/fits-cutout?ra='
            + str(ra)
            + '&dec='
            + str(dec)
            + '&pixscale='
            + str(slug.DECaLS_pixel_scale)
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
            + str(slug.DECaLS_pixel_scale)
            + '&layer='
            + layer
            + '&size='
            + str(size / slug.DECaLS_pixel_scale)
            + '&bands='
            + bands]


# Login NAOJ server
def login_naoj_server_3(config_path):
    ''' Runs well under python 2. In python 3, there's a widget 
    to enter username and password directly in Jupyter Notebook.
    In the configuration file, I wrote username in the first line, 
    and password in the second line.
    '''
    import urllib
    # Import HSC username and password
    config = Table.read(config_path, format='ascii.no_header')['col1']
    username = config[0]
    password = config[1]
    # Create a password manager
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

    ###### For downloading images ######
    # Add the username and password.
    top_level_url = 'https://hscdata.mtk.nao.ac.jp/das_quarry/dr2.1/'
    password_mgr.add_password(None, top_level_url, username, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib.request.build_opener(handler)

    # use the opener to fetch a URL
    opener.open(top_level_url)

    # Install the opener.
    # Now all calls to urllib2.urlopen use our opener.
    urllib.request.install_opener(opener)

    ###### For downloading PSFs ######
    # Add the username and password.
    top_level_url = 'https://hscdata.mtk.nao.ac.jp/psf/6/'
    password_mgr.add_password(None, top_level_url, username, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    # create "opener" (OpenerDirector instance)
    opener = urllib.request.build_opener(handler)

    # use the opener to fetch a URL
    opener.open(top_level_url)

    # Install the opener.
    # Now all calls to urllib2.urlopen use our opener.
    urllib.request.install_opener(opener)

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
########################## healpix related ##############################
#########################################################################
# I steal these functions from John Moustakas: https://github.com/moustakas/legacyhalos/blob/master/py/legacyhalos/misc.py
def radec2pix(nside, ra, dec):
    '''Convert `ra`, `dec` to nested pixel number.
    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.
    Returns:
        Array of integer pixel numbers using nested numbering scheme.
    Notes:
        This is syntactic sugar around::
            hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
        but also works with older versions of healpy that didn't have
        `lonlat` yet.
    '''
    import healpy as hp
    theta, phi = np.radians(90-dec), np.radians(ra)
    if np.isnan(np.sum(theta)) :
        raise ValueError("some NaN theta values")

    if np.sum((theta < 0)|(theta > np.pi))>0 :
        raise ValueError("some theta values are outside [0,pi]: {}".format(theta[(theta < 0)|(theta > np.pi)]))

    return hp.ang2pix(nside, theta, phi, nest=True)

def pix2radec(nside, pix):
    '''Convert nested pixel number to `ra`, `dec`.
    Args:
        nside (int): HEALPix `nside`, ``2**k`` where 0 < k < 30.
        ra (float or array): Right Accention in degrees.
        dec (float or array): Declination in degrees.
    Returns:
        Array of RA, Dec coorindates using nested numbering scheme. 
    Notes:
        This is syntactic sugar around::
            hp.pixelfunc.pix2ang(nside, pix, nest=True)
    
    '''
    import healpy as hp

    theta, phi = hp.pixelfunc.pix2ang(nside, pix, nest=True)
    ra, dec = np.degrees(phi), 90-np.degrees(theta)
    
    return ra, dec

def get_decals_subdir(nside, id_s16a, ra, dec, datadir):
    """Get the directory of DECaLS profiles/coadds from John Moustakas.
    
    Parameters:
        nsize (int): 'nside' used in healpix
        id_s16a (str): 'id_s16a' from Song's intermediate-z catalog
        datadir (str): directory of mother folder
    
    Returns:
        decals_dir (str): DECaLS directory
    
    """
    pixnum = radec2pix(nside, ra, dec)
    subdir = os.path.join(str(pixnum), str(id_s16a))
    decals_dir = os.path.abspath(os.path.join(datadir, str(nside), subdir))
    return decals_dir