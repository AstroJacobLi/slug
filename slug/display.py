# Import packages
from __future__ import division, print_function
import os
import copy

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

import sep

from .h5file import str2dic
from .profile import skyobj_value
from slug import imutils

from kungpao import imtools
from kungpao import io
from kungpao.display import display_single, IMG_CMAP, SEG_CMAP
from kungpao.galsbp import galSBP

__all__ = ["display_isophote", "SBP_single"]

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
    circle: **list** of floats. Maximun length is 3.

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

    cen_x, cen_y = int(img.shape[0]/2), int(img.shape[1]/2)

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
            e = Ellipse(xy=(iso['y0'], iso['x0']),
                        height=iso['sma'] * 2.0,
                        width=iso['sma'] * 2.0 * (1.0 - iso['ell']),
                        angle=iso['pa_norm'])
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
    linewidth=3, labelsize=25, ticksize=30, label='SBP', labelloc='lower left'):

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
    phys_size = imutils.phys_size(redshift,is_print=False)

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
        leg = ax1.legend(fontsize=labelsize, frameon=False, loc=labelloc)
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower, color=linecolor, alpha=0.3*alpha)
    
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)
    ax1.invert_yaxis()

    # Twin axis with linear scale
    if physical_unit and show_banner is True:
        ax4 = ax1.twiny() 
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
        lin_pos = [i**0.25 for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=ticksize)
        ax4.xaxis.set_label_coords(1, 1.025)

        ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize)

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
    x_min=1.0, x_max=4.0, alpha=1.0, r_interval=(20, 50), physical_unit=False, show_pa=True, show_banner=True,
    show_dots=False, show_grid=False, show_hline=True, vertical_line=None, linecolor='firebrick', linestyle='-', 
    linewidth=3, label=None, labelloc='lower left'):
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
    show_pa: boolean. If False, the figure will not show Position Angle panel.
    show_banner: boolean. If true, it will show the linear physical scale on the top of figure.
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
        if not show_pa:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.65])
            ax2 = fig.add_axes([0.08, 0.72, 0.85, 0.35])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
        else:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.48])
            ax2 = fig.add_axes([0.08, 0.55, 0.85, 0.20])
            ax3 = fig.add_axes([0.08, 0.75, 0.85, 0.20])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
            ax3.tick_params(direction='in')
    else:
        if not show_pa and len(ax) > 2:
            raise SyntaxError('The length of ax should be 2.')
            return None
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        if show_pa:
            ax3 = ax[2]
            ax3.tick_params(direction='in')

    # Calculate physical size
    phys_size = imutils.phys_size(redshift, is_print=False)
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
        leg = ax1.legend(fontsize=25, frameon=False, loc=labelloc)
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

    if show_pa:
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
            
        if physical_unit and show_banner:
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
    else:
        if physical_unit and show_banner:
            ax4 = ax2.twiny() 
            ax4.tick_params(direction='in')
            lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
            lin_pos = [i**0.25 for i in lin_label]
            ax4.set_xticks(lin_pos)
            ax4.set_xlim(ax2.get_xlim())
            ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
            ax4.xaxis.set_label_coords(1, 1.05)
            ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
            for tick in ax4.xaxis.get_major_ticks():
                tick.label.set_fontsize(25)

    # Other ornaments
    if vertical_line:
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            ax2.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            if show_pa:
                ax3.axvline(x=pos**0.25, ymin=0, ymax=1, 
                            color='gray', linestyle=style_list[k], linewidth=3)
    if show_grid:
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
        ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
        if show_pa:
            ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ax is None:
        return fig
    if show_pa:
        return ax1, ax2, ax3
    else:
        return ax1, ax2


# You can plot 1-D SBP using this, containing SBP, PA and eccentricity.
def SBP_shape_abspa(ell_free, ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0,
    x_min=1.0, x_max=4.0, parange=(10, 20), alpha=1.0, r_interval=(20, 50), physical_unit=False, 
    show_pa=True, show_banner=True, show_dots=False, show_grid=False, show_hline=True, 
    vertical_line=None, linecolor='firebrick', linestyle='-', linewidth=3, label=None):
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
    show_pa: boolean. If False, the figure will not show Position Angle panel.
    show_banner: boolean. If true, it will show the linear physical scale on the top of figure.
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
        if not show_pa:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.65])
            ax2 = fig.add_axes([0.08, 0.72, 0.85, 0.35])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
        else:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.48])
            ax2 = fig.add_axes([0.08, 0.55, 0.85, 0.20])
            ax3 = fig.add_axes([0.08, 0.75, 0.85, 0.20])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
            ax3.tick_params(direction='in')
    else:
        if not show_pa and len(ax) > 2:
            raise SyntaxError('The length of ax should be 2.')
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        if show_pa:
            ax3 = ax[2]
            ax3.tick_params(direction='in')

    # Calculate physical size
    phys_size = imutils.phys_size(redshift, is_print=False)
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

    if show_pa:
        # Position Angle profile
        from kungpao import utils
        
        y = np.abs(ell_free['pa_norm'] - np.nanmean(ell_free['pa_norm'][parange[0]:parange[1]]))

        pa_err = np.array([utils.normalize_angle(pa, lower=-90, 
                                                 upper=90, b=True) for pa in ell_free['pa_err']])
        if show_dots is True:
            ax3.errorbar((x ** 0.25), y, yerr=pa_err,
                         color='k', alpha=0.4, fmt='o', capsize=4, capthick=2, elinewidth=2)
        ax3.fill_between(x**0.25, y + pa_err, y - pa_err,
                         color=linecolor, alpha=0.3*alpha)
        ax3.plot(x**0.25, y, color=linecolor, linewidth=linewidth, alpha=alpha)
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
            
        if physical_unit and show_banner:
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
    else:
        if physical_unit and show_banner:
            ax4 = ax2.twiny()
            ax4.tick_params(direction='in')
            lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
            lin_pos = [i**0.25 for i in lin_label]
            ax4.set_xticks(lin_pos)
            ax4.set_xlim(ax2.get_xlim())
            ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
            ax4.xaxis.set_label_coords(1, 1.05)
            ax4.set_xticklabels([r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
            for tick in ax4.xaxis.get_major_ticks():
                tick.label.set_fontsize(25)

    # Other ornaments
    if vertical_line:
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            ax2.axvline(x=pos**0.25, ymin=0, ymax=1, 
                        color='gray', linestyle=style_list[k], linewidth=3)
            if show_pa:
                ax3.axvline(x=pos**0.25, ymin=0, ymax=1, 
                            color='gray', linestyle=style_list[k], linewidth=3)
    if show_grid:
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
        ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
        if show_pa:
            ax3.grid(linestyle='--', alpha=0.4, linewidth=2)

    if ax is None:
        return fig
    if show_pa:
        return ax1, ax2, ax3
    else:
        return ax1, ax2


# Plot SBP together, and also plot median profile
def SBP_stack(obj_cat, band, filenames, pixel_scale, zeropoint, ax=None, physical_unit=False, 
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, show_single=True, 
    vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', linewidth=5,
    single_alpha=0.3, single_color='firebrick', single_style='-', single_width=1, label=None, 
    single_label="S18A\ sky\ objects"):
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
        info = str2dic(f['info'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_free = f['ell_free'][band].value
        ell_fix = f['ell_fix'][band].value
        if sky_cat is None:
            off_set = 0.0
        else:
            off_set = skyobj_value(sky_cat,
                                        ra,
                                        dec,
                                        matching_radius=matching_radius,
                                        aperture=aperture,
                                        maxiters=5,
                                        showmedian=False)
        if k == 0:
            single_label = single_label
        else:
            single_label = None
        if show_single:
            SBP_single(
                ell_fix,
                redshift,
                pixel_scale,
                zeropoint,
                ax=ax1,
                offset=-off_set,
                physical_unit=physical_unit,
                x_min=x_min,
                x_max=x_max,
                show_banner=(k==0),
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)

        x = ell_fix['sma'] * pixel_scale * imutils.phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
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
        yerr_set = np.array([np.std(bootstrap(bootarr, 100, bootfunc=np.nanmedian)) for bootarr in y_stack.T])

    y = -2.5*np.log10(np.nanmedian(y_stack, axis=0)/(pixel_scale)**2) + zeropoint
    y_upper = -2.5*np.log10((np.nanmedian(y_stack, axis=0) + yerr_set)/(pixel_scale)**2) + zeropoint
    y_lower = -2.5*np.log10((np.nanmedian(y_stack, axis=0) - yerr_set)/(pixel_scale)**2) + zeropoint
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
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, show_pa=True, ninterp=30,
    parange=(10, 20), show_single=True, vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', 
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
    show_pa: boolean. If False, the figure will not show 'Position Angle' panel.
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
        if not show_pa:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.65])
            ax2 = fig.add_axes([0.08, 0.72, 0.85, 0.35])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
        else:
            ax1 = fig.add_axes([0.08, 0.07, 0.85, 0.48])
            ax2 = fig.add_axes([0.08, 0.55, 0.85, 0.20])
            ax3 = fig.add_axes([0.08, 0.75, 0.85, 0.20])
            ax1.tick_params(direction='in')
            ax2.tick_params(direction='in')
            ax3.tick_params(direction='in')
    else:
        if not show_pa and len(ax) > 2:
            raise SyntaxError('The length of ax should be 2.')
        ax1 = ax[0]
        ax2 = ax[1]
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        if show_pa:
            ax3 = ax[2]
            ax3.tick_params(direction='in')


    for k, obj in enumerate(obj_cat):
        # Load files
        f = h5py.File(filenames[k], 'r')
        # Load info
        info = str2dic(f['info'].value)
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
            off_set = skyobj_value(sky_cat,
                                        ra,
                                        dec,
                                        matching_radius=matching_radius,
                                        aperture=aperture,
                                        maxiters=5,
                                        showmedian=False)
        if k == 0:
            single_label = None #"S18A\ sky\ objects"
        else:
            single_label = None

        if physical_unit is False:
            raise ValueError('You must use physical sacle.')

        if show_pa:
                input_ax = [ax1, ax2, ax3]
        else:
            input_ax = [ax1, ax2]

        if show_single:
            SBP_shape_abspa(
                ell_free,
                ell_fix,
                redshift,
                pixel_scale,
                zeropoint,
                ax=input_ax,
                physical_unit=physical_unit,
                x_min=x_min,
                x_max=x_max,
                parange=parange,
                offset=-off_set,
                show_pa=show_pa,
                show_banner=(k==0),
                show_hline=False,
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)
        
        x_input = np.linspace(x_min, x_max, ninterp)

        # Interpolate for surface brightness
        x = ell_fix['sma'] * pixel_scale * imutils.phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens']-off_set, kind='cubic', fill_value='extrapolate')
        if k == 0:
            SB_stack = func(x_input)
        else:
            SB_stack = np.vstack((SB_stack, func(x_input)))

        # Interpolate for ellipticity
        x = ell_free['sma'] * HSC_pixel_scale * imutils.phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['ell'])
        func = interpolate.interp1d(x[mask]**0.25, ell_free['ell'][mask], kind='cubic', fill_value='extrapolate')
        if k == 0:
            e_stack = func(x_input)
        else:
            e_stack = np.vstack((e_stack, func(x_input)))
        
        # Interpolate for position angle
        x = ell_free['sma'] * HSC_pixel_scale * imutils.phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['pa_norm'])

        func = interpolate.interp1d(x[mask]**0.25, 
            abs(ell_free['pa_norm']-np.nanmean(ell_free['pa_norm'][parange[0]:parange[1]]))[mask], 
            kind='cubic', fill_value='extrapolate')
        if k == 0:
            pa_stack = func(x_input)
        else:
            pa_stack = np.vstack((pa_stack, func(x_input)))
        
        f.close()

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        SB_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in SB_stack.T])
        e_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in e_stack.T])
        pa_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in pa_stack.T])
    
    # ax1: SBP
    y = -2.5*np.log10(np.nanmedian(SB_stack, axis=0)/(pixel_scale)**2) + zeropoint
    y_upper = -2.5*np.log10((np.nanmedian(SB_stack, axis=0) + SB_err)/(pixel_scale)**2) + zeropoint
    y_lower = -2.5*np.log10((np.nanmedian(SB_stack, axis=0) - SB_err)/(pixel_scale)**2) + zeropoint
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
    
    if show_pa:
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


    # Return
    if ax is None:
        return fig, SB_stack, e_stack, pa_stack, x_input
    return input_ax, SB_stack, e_stack, pa_stack, x_input


def SBP_thick_line(stack_set, pixel_scale, zeropoint, ax=None, 
    x_min=1.0, x_max=4.0, show_pa=True, show_banner=True, parange=(10, 20), ninterp=30, vertical_line=None, 
    ismedian=True, linecolor=['brown', 'blue', 'green'], linestyle=['-', '--', '-.'],
    fillcolor=['orange', 'skyblue', 'seagreen'], linewidth=5, label=None):
    '''
    stack_set: SB, e, PA, x_input
    '''
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext
    # Import axes
    if not show_pa and len(ax) > 2:
        raise SyntaxError('The length of ax should be 2.')
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.tick_params(direction='in')
    ax2.tick_params(direction='in')
    if show_pa:
        ax3 = ax[2]
        ax3.tick_params(direction='in')

    # Plot lines
    for k, stack in enumerate(stack_set):
        # Calculate error
        with NumpyRNGContext(2333):
            if ismedian:
                btfunc = np.nanmedian
            else:
                btfunc = np.nanmean
            SB_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[0].T])
            e_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[1].T])
            pa_err = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[2].T])
            
        x_input = stack[3]
        # ax1: SBP
        y = -2.5*np.log10(np.nanmedian(stack[0], axis=0)/(pixel_scale)**2) + zeropoint
        y_upper = -2.5*np.log10((np.nanmedian(stack[0], axis=0) + SB_err)/(pixel_scale)**2) + zeropoint
        y_lower = -2.5*np.log10((np.nanmedian(stack[0], axis=0) - SB_err)/(pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]

        if label is not None:
            ax1.plot(x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k],
                 label=r'$\mathrm{' + label[k] + '}$', alpha=1)
            leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
            for l in leg.legendHandles:
                l.set_alpha(1)
        else:
            ax1.plot(x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
        ax1.fill_between(x_input, y_upper, y_lower, color=fillcolor[k], alpha=0.4)
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
        y = np.nanmedian(stack[1], axis=0)
        ax2.plot(x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
        ax2.fill_between(x_input, y - e_err, y + e_err, color=fillcolor[k], alpha=0.4)
        ax2.xaxis.set_major_formatter(NullFormatter())
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        ax2.set_xlim(x_min, x_max)
        ylim = ax2.get_ylim()
        ax2.set_ylabel(r'$e$', fontsize=35)
        ytick_pos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ax2.set_yticks(ytick_pos)
        ax2.set_yticklabels([r'$'+str(round(i,2))+'$' for i in ytick_pos])
        ax2.set_ylim(ylim)
        
        if show_pa:
            # ax3: Position angle
            y = np.nanmedian(stack[2], axis=0)
            ax3.plot(x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
            ax3.fill_between(x_input, y - pa_err, y + pa_err, color=fillcolor[k], alpha=0.4)
            ax3.xaxis.set_major_formatter(NullFormatter())
            for tick in ax3.yaxis.get_major_ticks():
                tick.label.set_fontsize(25)
            ax3.set_xlim(x_min, x_max)
            ylim = ax3.get_ylim()
            ax3.set_ylabel(r'$\mathrm{PA\ [deg]}$', fontsize=25)
            ytick_pos = [0, 15, 30, 45, 60]
            ax3.set_yticks(ytick_pos)
            ax3.set_ylim(ylim)

        if show_banner and show_pa:
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
        if show_banner and not show_pa:
            ax4 = ax2.twiny()
            ax4.tick_params(direction='in')
            lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
            lin_pos = [i**0.25 for i in lin_label]
            ax4.set_xticks(lin_pos)
            ax4.set_xlim(ax2.get_xlim())
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
            if show_pa:
                ax3.axvline(x=pos**0.25, ymin=0, ymax=1, 
                            color='gray', linestyle=style_list[k], linewidth=3)
    if show_pa:
        return [ax1, ax2, ax3]
    else:
        return [ax1, ax2]

        