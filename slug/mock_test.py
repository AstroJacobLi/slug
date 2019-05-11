from __future__ import division, print_function

import numpy as np

from astropy import wcs
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

from kungpao.display import display_single, IMG_CMAP, SEG_CMAP

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
    for j, i in itertools.product(*map(range, array.shape)):
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
    #allowError = 0.5
    #assert (array.sum() < result.sum() * (1+allowError)) & (array.sum() >result.sum() * (1-allowError))
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
    obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))
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
def tractor_iteration(obj_cat, w, img_data, invvar, psf_obj, pixel_scale, shape_method='manual', 
                      kfold=4, first_num=50, fig_name=None):
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
    shape_method: if 'manual', then adopt manually measured shape. If 'decals', then adopt DECaLS shape from tractor files.
    kfold: int, iteration time.
    first_num: how many objects will be fit in the first run.
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

    step = int((len(obj_cat) - first_num)/(kfold-1))
    for i in range(kfold):
        if i == 0:
            obj_small_cat = obj_cat[:first_num]
            sources = add_tractor_sources(obj_small_cat, None, w, shape_method='manual')
        else:
            obj_small_cat = obj_cat[first_num + step*(i-1) : first_num + step*(i)]
            sources = add_tractor_sources(obj_small_cat, sources, w, shape_method='manual')

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
        plt.rc('font', size=20)
        if i % 2 == 1 or i == (kfold-1) :
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18,8))

            trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[:])

            ax1 = display_single(img_data, ax=ax1, scale_bar=False)
            ax1.set_title('raw image')
            ax2 = display_single(trac_mod_opt, ax=ax2, scale_bar=False, contrast=0.02)
            ax2.set_title('tractor model')
            ax3 = display_single(abs(img_data - trac_mod_opt), ax=ax3, scale_bar=False, color_bar=True, contrast=0.05)
            ax3.set_title('residual')

            if i == (kfold-1):
                if fig_name is not None:
                    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
                    plt.show()
                    print('The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))))
            else:
                plt.show()
                print('The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data)) 

        #trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[1:])
        #ax4 = display_single(img_data - trac_mod_opt, ax=ax4, scale_bar=False, color_bar=True, contrast=0.05)
        #ax4.set_title('remain central galaxy')


    return sources, trac_obj, fig