import os
import slug
import numpy as np
import matplotlib.pyplot as plt
from kungpao import io
from kungpao.display import display_single, IMG_CMAP, SEG_CMAP
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column

def run_intermediate_sample(num, z_set, filename, img_path):
    prefix = 'HSC-mid-' + str(num)
    img = fits.open(img_path + filename[num])[1]
    w = wcs.WCS(img)
    display_single(img.data)
    plt.show()
    # phys_size
    redshift = z_set[num]['z_best']
    phys_size = slug.phys_size(redshift)
    # extract_obj
    data = img.data.byteswap().newbyteorder()
    objects, segmap = slug.extract_obj(
        data,
        b=30,
        f=5,
        pixel_scale=slug.HSC_pixel_scale,
        deblend_cont=0.1,
        deblend_nthresh=20,
        show_fig=False)
    # make mask
    seg_mask = slug.make_binary_mask(data, w, segmap, show_fig=False)
    # evaluate_sky
    bkg_global = slug.evaluate_sky(data, show_fig=False, show_hist=False)
    # Save image and mask
    if not os.path.isdir('Images'):
        os.mkdir('Images')
    if not os.path.isdir('Masks'):
        os.mkdir('Masks')
    img_fits = './Images/' + prefix + '_img.fits'
    msk_fits = './Masks/' + prefix + '_msk.fits'
    
    io.save_to_fits(data, img_fits, wcs=w)
    io.save_to_fits(seg_mask.astype('uint8'), msk_fits, wcs=w)
    display_single(data*(~seg_mask))
    plt.show()
    # Run ELLIPSE
    iraf_path = '/Users/jiaxuanli/Research/slug/slug/iraf/macosx/'
    ell_free, ell_fix = slug.run_SBP(
        img_fits,
        msk_fits,
        slug.HSC_pixel_scale,
        phys_size,
        iraf_path,
        step=0.1,
        n_clip=3,
        low_clip=3.0,
        upp_clip=2.5,
        outPre=prefix)