# Import packages
from __future__ import division, print_function
import os
import copy

import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.table import Table

from astropy.visualization import (ZScaleInterval,
                                   AsymmetricPercentileInterval)
from astropy.visualization import make_lupton_rgb

from matplotlib import colors
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.sequential import (Greys_9,
                                               OrRd_9,
                                               Blues_9,
                                               Purples_9,
                                               YlGn_9)

import sep

from .h5file import str2dic
from .imutils import phys_size
import slug
from slug import imutils

from kungpao import imtools
from kungpao import io
from kungpao.display import IMG_CMAP, SEG_CMAP

__all__ = ["display_single", "display_isophote", "SBP_single"]

#########################################################################
############################ Display related ###########################
#########################################################################


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   text_fontsize=30,
                   text_y_offset=0.80,
                   text_color='w'):
    """Display single image. From `kungpao`.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)

        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        ax1.text(text_x_0, text_y_0,
                 r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def _display_single(img,
                    pixel_scale=0.168,
                    physical_scale=None,
                    xsize=8,
                    ysize=8,
                    ax=None,
                    stretch='arcsinh',
                    scale='zscale',
                    scale_manual=None,
                    contrast=0.25,
                    no_negative=False,
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                    cmap=IMG_CMAP,
                    scale_bar=True,
                    scale_bar_length=5.0,
                    scale_bar_fontsize=20,
                    scale_bar_y_offset=0.5,
                    scale_bar_color='w',
                    scale_bar_loc='left',
                    color_bar=False,
                    color_bar_loc=1,
                    color_bar_width='75%',
                    color_bar_height='5%',
                    color_bar_fontsize=18,
                    color_bar_color='w',
                    add_text=None,
                    text_fontsize=30,
                    text_y_offset=0.80,
                    text_color='w'):
    """Display single image. From `kungpao`.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    if scale_manual is not None:
        assert len(scale_manual) == 2, '# length of manual scale must be two!'
        zmin, zmax = scale_manual

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)

        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        ax1.text(text_x_0, text_y_0,
                 r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig, zmin, zmax
    return ax1, zmin, zmax


def display_multiple(data_array, text=None, ax=None, **kwargs):
    if ax is None:
        fig, axes = plt.subplots(
            1, len(data_array), figsize=(len(data_array) * 4, 8))
    else:
        axes = ax

    if text is None:
        _, zmin, zmax = _display_single(data_array[0], ax=axes[0], **kwargs)
    else:
        _, zmin, zmax = _display_single(
            data_array[0], add_text=text[0], ax=axes[0], **kwargs)
    for i in range(1, len(data_array)):
        if text is None:
            _display_single(data_array[i], ax=axes[i], scale_manual=[
                            zmin, zmax], scale_bar=False, **kwargs)
        else:
            _display_single(data_array[i], add_text=text[i], ax=axes[i], scale_manual=[
                            zmin, zmax], scale_bar=False, **kwargs)

    plt.subplots_adjust(wspace=0.0)
    if ax is None:
        return fig
    else:
        return axes


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
            e = Ellipse(xy=(iso['x0'], iso['y0']),
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * \
            np.log10((ell_fix['intens'].data + offset) /
                     (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix[intens_err_name]) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix[intens_err_name]) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * \
            np.log10((ell_fix['intens'].data + offset) /
                     (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix[intens_err_name]) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix[intens_err_name]) / (pixel_scale) ** 2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    # If `nan` at somewhere, interpolate `nan`.
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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.3*alpha, label=None)

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize)

    plt.sca(ax1)

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

# You can plot 1-D SBP using this, without plotting the PA and eccentricity.


def SBP_single_linear(ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0,
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        # y is in the unit of \muJy/arcsec^2
        y = 3.631 * (ell_fix['intens'].data + offset) / (pixel_scale)**2 / \
            10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2
        y_upper = 3.631 * (ell_fix['intens'] + offset + ell_fix[intens_err_name]) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        y_lower = 3.631 * (ell_fix['intens'] + offset - ell_fix[intens_err_name]) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        upper_yerr = y_upper - y
        lower_yerr = y - y_lower
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{\mu Jy/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        # y is in the unit of \muJy/arcsec^2
        y = 3.631 * (ell_fix['intens'].data + offset) / (pixel_scale)**2 / \
            10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2
        y_upper = 3.631 * (ell_fix['intens'] + offset + ell_fix[intens_err_name]) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        y_lower = 3.631 * (ell_fix['intens'] + offset - ell_fix[intens_err_name]) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        upper_yerr = y_upper - y
        lower_yerr = y - y_lower
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{\mu Jy/arcsec^2}]$'

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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.3*alpha, label=None)

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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

# You can plot 1-D SBP using this, without plotting the PA and eccentricity.


def SBP_single_try(ell_fix, redshift, pixel_scale, zeropoint, ax=None, offset=0.0,
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * \
            np.log10((ell_fix['intens'].data + offset) /
                     (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix[intens_err_name]) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix[intens_err_name]) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * \
            np.log10((ell_fix['intens'].data + offset) /
                     (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix[intens_err_name]) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix[intens_err_name]) / (pixel_scale) ** 2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    # If `nan` at somewhere, interpolate `nan`.
    nanidx = np.where(np.isnan(y))[0]
    if len(nanidx) > 1:
        from sklearn.cluster import KMeans
        X = np.array(list(zip(nanidx, np.zeros_like(nanidx))))
        kmeans = KMeans(n_clusters=2).fit(X)
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        if (max(centroids[:, 0]) - min(centroids[:, 0]) < 3) and np.ptp(nanidx[labels == 0]) > 2:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value='extrapolate')
            y[nanidx[labels == 0]] = func(x[nanidx[labels == 0]]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            y_upper[nanidx[0]:] = np.nan
            y_lower[nanidx[0]:] = np.nan
    elif len(nanidx) == 1:
        try:
            if abs(y[nanidx - 1] - y[nanidx + 1]) < 0.5:
                print('interpolate NaN')
                from scipy.interpolate import interp1d
                mask = (~np.isnan(y))
                func = interp1d(x[mask]**0.25, y[mask],
                                kind='cubic', fill_value='extrapolate')
                y[nanidx] = func(x[nanidx]**0.25)
            else:
                y[nanidx[0]:] = np.nan
                y_upper[nanidx[0]:] = np.nan
                y_lower[nanidx[0]:] = np.nan
        except:
            print('')
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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.3*alpha, label=None)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)
    # ax1.invert_yaxis()

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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


def load_SBP(obj, survey='HSC', band='r', sky_cat=None, matching_radius=[1, 4], aperture='84'):
    '''
    Load surface brightness profile of a given object. 

    For HSC: sky objects are subtracted from ellipse intensity.
    For DECaLS: color correction is not applied. 

    Parameters:
        obj: object in catalog
        survey (str): either "HSC" or "DECaLS"
        band (str): such as "r"
        sky_cat: SkyObject catalog. Only use for HSC.
        matching_radius: float, in arcmin. We match sky objects around the given object within this radius. Only use for HSC.
        aperture: string, must be in the `SkyObj_aperture_dic`. Only use for HSC.

    Returns:
        ell_fix (astropy.table.Table)
    '''

    import h5py
    from .imutils import skyobj_value

    if survey == 'HSC':
        # Load files
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix = Table(f[f'{band}-band']['ell_fix'].value)
        f.close()

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
        ell_fix['intens'] += off_set

    elif survey == 'DECaLS':
        # Load files
        ellipsefit = Table.read(obj['decals_dir'])
        ell_fix = Table(
            data=[
                ellipsefit[f'{band.upper()}_SMA'].data[0],  # pixel
                ellipsefit[f'{band.upper()}_INTENS'].data[0] * (slug.DECaLS_pixel_scale) **
                2,  # nanomaggie/pixel
                ellipsefit[f'{band.upper()}_INTENS_ERR'].data[0] * (slug.DECaLS_pixel_scale) **
                2,  # nanomaggie/pixel
                np.ones_like(ellipsefit[f'{band.upper()}_SMA'].data[0]
                             ) * ellipsefit['EPS'].data[0],  # ellipticity
                np.ones_like(ellipsefit[f'{band.upper()}_SMA'].data[0]
                             ) * ellipsefit['PA'].data[0],  # PA
            ],
            names=['sma', 'intens', 'intens_err', 'ell', 'pa'])  # r-band ellipse result
        redshift = obj['z_best']

    else:
        raise ValueError("Only support HSC or DECaLS!")

    return ell_fix


def gen_median_SBP(x_input, y_stack, zeropoint, pixel_scale):
    from astropy.stats import bootstrap

    nan_ratio = np.sum(np.isnan(y_stack), axis=0) / len(y_stack)
    nan_flag = (nan_ratio > 0.6)

    y = -2.5 * np.log10(np.nanmedian(y_stack, axis=0) /
                        (pixel_scale)**2) + zeropoint
    yerr_set = np.array(
        [np.std(bootstrap(bootarr, 200, bootfunc=np.nanmedian)) for bootarr in y_stack.T])
    y_upper = -2.5 * \
        np.log10((np.nanmedian(y_stack, axis=0) + yerr_set) /
                 (pixel_scale)**2) + zeropoint
    y_lower = -2.5 * \
        np.log10((np.nanmedian(y_stack, axis=0) - yerr_set) /
                 (pixel_scale)**2) + zeropoint

    nan_flag |= np.isnan(y + y_upper + y_lower)

    y[nan_flag] = np.nan
    yerr_set[nan_flag] = np.nan
    y_upper[nan_flag] = np.nan
    y_lower[nan_flag] = np.nan

    return x_input, y, [y_lower, y_upper, yerr_set]


def plot_median_SBP(x_input, y_stack, zeropoint, pixel_scale, ax, ls='-'):
    from astropy.stats import bootstrap

    x_input, y, [y_lower, y_upper, yerr_set] = gen_median_SBP(
        x_input, y_stack, zeropoint, pixel_scale)
    ax.plot(x_input, y, color='k', linewidth=4, linestyle=ls,
            label=r'$\mathrm{Median}$', alpha=0.8, zorder=10)
    ax.fill_between(x_input, y_upper, y_lower,
                    color='gray', alpha=0.9, zorder=10)
    return ax


def plot_median_diff(HSC_SB_stack, DECaLS_SB_stack, ax=None,
                     linecolor='brown', linewidth=5, linestyle='-',
                     label=None, labelloc='upper right',
                     labelsize=20, ticksize=30, show_banner=True):

    x_input = np.arange(1.0, 5.5, 0.05)
    x_input, y_HSC, [y_lower_HSC, y_upper_HSC, yerr_HSC] = gen_median_SBP(
        x_input, HSC_SB_stack, slug.HSC_zeropoint, slug.HSC_pixel_scale)

    x_input, y_DECaLS, [y_lower_DECaLS, y_upper_DECaLS, yerr_DECaLS] = gen_median_SBP(
        x_input, DECaLS_SB_stack, slug.DECaLS_zeropoint, slug.DECaLS_pixel_scale)

    # Plot!
    y = y_HSC - y_DECaLS
    yerr = np.sqrt(yerr_HSC**2 + yerr_DECaLS**2)
    yerr[yerr > 0.3] = np.nan

    ax1 = ax
    ax1.tick_params(direction='in')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    # remove discontinuity
    mask = (np.isnan(y))
    y[mask] = np.nan
    yerr[mask] = np.nan

    if label is not None:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle=linestyle,
                 label=r'$\mathrm{' + label + '}$', alpha=1, zorder=10)
        leg = ax1.legend(fontsize=20, frameon=False, loc=labelloc)
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=1, zorder=10)
    ax1.fill_between(x_input, y + yerr, y - yerr,
                     color='gray', alpha=0.9, zorder=10)

    y_lim = [-1.1, 1.1]
    #x_lim = [3.0, 4.5]
    ax1.set_ylim(y_lim)
    # ax1.set_xlim(x_lim)

    # Add text
    # ax1.text(3.75, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.1,
    #        r'$\mathrm{' + label + '}$',
    #        horizontalalignment='center', verticalalignment='center', fontsize=20,
    #        bbox=dict(facecolor='wheat', edgecolor='k', boxstyle='round, pad=0.4'))

    # ax.yaxis.set_major_formatter(FormatStrFormatter(r'$%.2f$'))
    #ax.vlines(100**0.25, min(y_lim), max(y_lim), color='gray', linestyle='--', linewidth=2)
    #ax1.hlines(0, x_lim[0], x_lim[1], color='k', linestyle='--', linewidth=2, zorder=11)

    xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
    ylabel = r'$\mu_{\mathrm{HSC}} - \mu_{\mathrm{DECaLS}}$' + \
        '\n' + '$[\mathrm{mag/arcsec^2}]$'
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)

    if show_banner:
        ax4 = ax1.twiny()
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
        lin_pos = [i**0.25 for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=ticksize)
        ax4.xaxis.set_label_coords(1.0, 1.025)

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize)
    # Return
    if ax is None:
        return fig
    return ax1


def SBP_single_upper_limit(ell_fix, redshift, pixel_scale, zeropoint, skyval=0.0, skystd=0.0, filter_corr=0,
                           ax=None, x_min=1.0, x_max=4.0, ylim=None, alpha=1, physical_unit=False, show_dots=False, show_grid=False,
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * np.log10((ell_fix['intens'].data - skyval) /
                            (pixel_scale)**2) + zeropoint + filter_corr
        y_upper = -2.5 * np.log10((ell_fix['intens'] - skyval + ell_fix[intens_err_name]) / (
            pixel_scale)**2) + zeropoint + filter_corr
        y_lower = -2.5 * np.log10((ell_fix['intens'] - skyval - ell_fix[intens_err_name]) / (
            pixel_scale)**2) + zeropoint + filter_corr
        y_sky_upper = -2.5 * np.log10((ell_fix['intens'] - skyval + ell_fix[intens_err_name] + skystd) / (
            pixel_scale)**2) + zeropoint + filter_corr
        y_sky_lower = -2.5 * np.log10((ell_fix['intens'] - skyval - ell_fix[intens_err_name] - skystd) / (
            pixel_scale)**2) + zeropoint + filter_corr
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'].data - skyval) /
                            (pixel_scale)**2) + zeropoint + filter_corr
        y_upper = -2.5 * np.log10((ell_fix['intens'] - skyval + ell_fix[intens_err_name]) / (
            pixel_scale) ** 2) + zeropoint + filter_corr
        y_lower = -2.5 * np.log10((ell_fix['intens'] - skyval - ell_fix[intens_err_name]) / (
            pixel_scale) ** 2) + zeropoint + filter_corr
        y_sky_upper = -2.5 * np.log10((ell_fix['intens'] - skyval + ell_fix[intens_err_name] + skystd) / (
            pixel_scale)**2) + zeropoint + filter_corr
        y_sky_lower = -2.5 * np.log10((ell_fix['intens'] - skyval - ell_fix[intens_err_name] - skystd) / (
            pixel_scale)**2) + zeropoint + filter_corr
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    # If `nan` at somewhere, interpolate `nan`.
    nanidx = np.where(np.isnan(y))[0]
    if len(nanidx) > 1:
        from sklearn.cluster import KMeans
        X = np.array(list(zip(nanidx, np.zeros_like(nanidx))))
        kmeans = KMeans(n_clusters=2).fit(X)
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        if (max(centroids[:, 0]) - min(centroids[:, 0]) < 3) and np.ptp(nanidx[labels == 0]) > 2:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value='extrapolate')
            y[nanidx[labels == 0]] = func(x[nanidx[labels == 0]]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            y_upper[nanidx[0]:] = np.nan
            y_lower[nanidx[0]:] = np.nan
            y_sky_upper[nanidx[0]:] = np.nan
            y_sky_lower[nanidx[0]:] = np.nan

    elif len(nanidx) == 1:
        if nanidx + 1 > len(nanidx) or nanidx - 1 < 0:
            print('Sorry, cannot replace NaN')
        elif abs(y[nanidx - 1] - y[nanidx + 1]) < 0.5:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value=np.nan)
            y[nanidx] = func(x[nanidx]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            y_upper[nanidx[0]:] = np.nan
            y_lower[nanidx[0]:] = np.nan
            y_sky_upper[nanidx[0]:] = np.nan
            y_sky_lower[nanidx[0]:] = np.nan

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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.4*alpha, label=None)
    if ylim is None:
        ylim = ax1.get_ylim()

    for i in range(len(y_sky_lower)):
        if np.isnan(y_sky_lower[i]):
            y_sky_lower[i] = max(ylim)
    ax1.fill_between(x**0.25, y_sky_upper, y_sky_lower,
                     color=linecolor, alpha=0.13*alpha, label=None)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)
    # ax1.invert_yaxis()
    ax1.set_ylim(max(ylim), min(ylim))

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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

    '''
    # zoom-in linear view
    ax_ins = ax[1]
    mask = (x**0.25 > 2.3)
    flux = 3.631 * (ell_fix['intens'] - skyval) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)  #\muJy/arcsec^2
    flux_upper = 3.631 * (ell_fix['intens'] - skyval + ell_fix['int_err'] + skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)  #\muJy/arcsec^2
    flux_lower = 3.631 * (ell_fix['intens'] - skyval - ell_fix['int_err'] - skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)  #\muJy/arcsec^2
    ax[1].plot(x[mask]**0.25, flux[mask], color=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax[1].fill_between(x[mask]**0.25, flux_upper[mask], flux_lower[mask], color=linecolor, alpha=0.1*alpha, label=None)
    ax[1].set_xlabel(r'$(R/\mathrm{kpc})^{1/4}$', fontsize=12)
    ax[1].set_ylabel(r'$\mu\,[\mathrm{\mu Jy/arcsec^2}]$', fontsize=15)
    #plt.yticks(np.array([0, 0.1, 0.2, 0.3, 0.4]) * sum(n), ['0', '0.1', '0.2', '0.3', '0.4'], fontsize=10)
    #plt.xticks(np.arange(0,0.18,0.03), fontsize=10)
    ax[1].tick_params(direction='in', length=1)
    '''

    # Return
    if ax is None:
        return fig
    return ax1  # , ax_ins


# You can plot 1-D SBP using this, without plotting the PA and eccentricity.
def SBP_single_upper_limit_1error(ell_fix, redshift, pixel_scale, zeropoint, skyval=0.0, skystd=0.0, filter_corr=0,
                                  ax=None, x_min=1.0, x_max=4.0, ylim=None, alpha=1, physical_unit=False, show_dots=False, show_grid=False,
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        y = -2.5 * np.log10((ell_fix['intens'].data - skyval) /
                            (pixel_scale)**2) + zeropoint + filter_corr
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'].data - skyval) /
                            (pixel_scale)**2) + zeropoint + filter_corr
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'

    ellipse_err = ell_fix[intens_err_name].data
    ellipse_err[np.isnan(ellipse_err)] = 0.0
    err = np.sqrt(ellipse_err**2 + skystd**2)
    y_upper = -2.5 * \
        np.log10((ell_fix['intens'] - skyval + err) /
                 (pixel_scale)**2) + zeropoint + filter_corr
    y_lower = -2.5 * \
        np.log10((ell_fix['intens'] - skyval - err) /
                 (pixel_scale)**2) + zeropoint + filter_corr
    y_lower[np.isnan(y_lower)] = 35.0  # in case y_lower is nan
    #y_sky_upper = -2.5 * np.log10((ell_fix['intens'] - skyval + ell_fix[intens_err_name] + skystd) / (pixel_scale)**2) + zeropoint + filter_corr
    #y_sky_lower = -2.5 * np.log10((ell_fix['intens'] - skyval - ell_fix[intens_err_name] - skystd) / (pixel_scale)**2) + zeropoint + filter_corr
    upper_yerr = y_lower - y
    lower_yerr = y - y_upper
    asymmetric_error = [lower_yerr, upper_yerr]
    # print(asymmetric_error)

    # If `nan` at somewhere, interpolate `nan`.
    nanidx = np.where(np.isnan(y))[0]
    if len(nanidx) > 1:
        from sklearn.cluster import KMeans
        X = np.array(list(zip(nanidx, np.zeros_like(nanidx))))
        kmeans = KMeans(n_clusters=2).fit(X)
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        if (max(centroids[:, 0]) - min(centroids[:, 0]) < 3) and np.ptp(nanidx[labels == 0]) > 2:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value='extrapolate')
            y[nanidx[labels == 0]] = func(x[nanidx[labels == 0]]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            y_upper[nanidx[0]:] = np.nan
            y_lower[nanidx[0]:] = np.nan
            #y_sky_upper[nanidx[0]:] = np.nan
            #y_sky_lower[nanidx[0]:] = np.nan

    elif len(nanidx) == 1:
        if nanidx + 1 > len(nanidx) or nanidx - 1 < 0:
            print('Sorry, cannot replace NaN')
        elif abs(y[nanidx - 1] - y[nanidx + 1]) < 0.5:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value=np.nan)
            y[nanidx] = func(x[nanidx]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            y_upper[nanidx[0]:] = np.nan
            y_lower[nanidx[0]:] = np.nan
            #y_sky_upper[nanidx[0]:] = np.nan
            #y_sky_lower[nanidx[0]:] = np.nan

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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)
    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.4*alpha, label=None)
    if ylim is None:
        ylim = ax1.get_ylim()

    '''
    for i in range(len(y_sky_lower)):
        if np.isnan(y_sky_lower[i]):
            y_sky_lower[i] = max(ylim)
    ax1.fill_between(x**0.25, y_sky_upper, y_sky_lower, color=linecolor, alpha=0.13*alpha, label=None)
    '''
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)
    # ax1.invert_yaxis()
    ax1.set_ylim(max(ylim), min(ylim))

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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
    return ax1  # , ax_ins


# You can plot 1-D SBP using this, without plotting the PA and eccentricity.
def SBP_single_arcsinh(ell_fix, redshift, pixel_scale, zeropoint, skyval=0.0, skystd=0.0,
                       ax=None, offset=0.0,
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
    skycat: skyobj catalog
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
    phys_size = imutils.phys_size(redshift, is_print=False)

    # 1-D profile
    if 'intens_err' in ell_fix.colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'

    if physical_unit is True:
        x = ell_fix['sma'] * pixel_scale * phys_size
        # y is in the unit of \muJy/arcsec^2
        y = 3.631 * (ell_fix['intens'] - skyval) / (pixel_scale)**2 / \
            10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2
        y = np.arcsinh(y)
        y_upper = 3.631 * (ell_fix['intens'] - skyval + ell_fix[intens_err_name] +
                           skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        y_upper = np.arcsinh(y_upper)
        y_lower = 3.631 * (ell_fix['intens'] - skyval - ell_fix[intens_err_name] -
                           skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        y_lower = np.arcsinh(y_lower)
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mathrm{arcsinh}\,\mu\,[\mathrm{\mu Jy/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = 3.631 * (ell_fix['intens'] - skyval) / (pixel_scale)**2 / \
            10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2
        y_upper = 3.631 * (ell_fix['intens'] - skyval + ell_fix[intens_err_name] +
                           skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        y_lower = 3.631 * (ell_fix['intens'] - skyval - ell_fix[intens_err_name] -
                           skystd) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{arcsec})^{1/4}$'
        ylabel = r'$\mathrm{arcsinh}\,\mu\,[\mathrm{\mu Jy/arcsec^2}]$'

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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)

    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=0.3*alpha, label=None)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)

    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)
    ax1.set_ylim(-0.18, 0.78)

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

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
        for tick in ax4.xaxis.get_major_ticks():
            tick.label.set_fontsize(ticksize)

    # show magnitude on the right axis
    ax5 = ax1.twinx()
    ax5.tick_params(direction='in')
    lin_label = np.arange(25, 33, 1)
    lin_pos = [np.arcsinh(10**((22.5 - i)/2.5) * 3.631) for i in lin_label]
    ax5.set_yticks(lin_pos)
    ax5.set_ylim(ax1.get_ylim())
    ax5.set_ylabel(r'$\mu\,[\mathrm{mag/arcsec^2}]$', fontsize=ticksize)
    ax5.yaxis.set_label_coords(1.07, 0.5)
    ax5.set_yticklabels(
        [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
    for tick in ax5.xaxis.get_major_ticks():
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


# Plot SBP together, and also plot median profile
def mu_diff_figure_1error(ell_fix_other, ell_fix_fid, survey_other, survey_fid, redshift, plot_err=True,
                          skyval=[0.0, 0.0], skystd=[0.0, 0.0], zp=None, filter_corr=[0, 0],
                          ax=None, x_min=1.0, x_max=4.0, alpha=1.0, vertical_line=None, show_banner=True,
                          linecolor='brown', linewidth=3, label=None, ticksize=20,
                          labelsize=20, labelloc='lower right'):
    '''
    filter_corr[0] + mag(ell_fix_other) = SDSS_mag
    filter_corr[1] + mag(ell_fix_fid) = SDSS_mag

    skyval = [skyval_other, skyval_fid]
    skystd = [skystd_other, skystd_fid]
    '''
    def fill_defects(y, y_upper=None, y_lower=None):
        nanidx = np.where(np.isnan(y))[0]
        if len(nanidx) > 1:
            from sklearn.cluster import KMeans
            X = np.array(list(zip(nanidx, np.zeros_like(nanidx))))
            kmeans = KMeans(n_clusters=2).fit(X)
            labels = kmeans.predict(X)
            centroids = kmeans.cluster_centers_
            if (max(centroids[:, 0]) - min(centroids[:, 0]) < 3) and np.ptp(nanidx[labels == 0]) > 2:
                print('interpolate NaN')
                from scipy.interpolate import interp1d
                mask = (~np.isnan(y))
                func = interp1d(x[mask]**0.25, y[mask],
                                kind='cubic', fill_value='extrapolate')
                y[nanidx[labels == 0]] = func(x[nanidx[labels == 0]]**0.25)
            else:
                y[nanidx[0]:] = np.nan
                if y_upper is not None:
                    y_upper[nanidx[0]:] = np.nan
                    y_lower[nanidx[0]:] = np.nan
        elif len(nanidx) == 1:
            if nanidx + 1 > len(nanidx) or nanidx - 1 < 0:
                print('Sorry, cannot replace NaN')
            elif abs(y[nanidx - 1] - y[nanidx + 1]) < 0.5:
                print('interpolate NaN')
                from scipy.interpolate import interp1d
                mask = (~np.isnan(y))
                func = interp1d(x[mask]**0.25, y[mask],
                                kind='cubic', fill_value=np.nan)
                y[nanidx] = func(x[nanidx]**0.25)
            else:
                y[nanidx[0]:] = np.nan
                if y_upper is not None:
                    y_upper[nanidx[0]:] = np.nan
                    y_lower[nanidx[0]:] = np.nan

        if y_upper is not None:
            return [y, y_upper, y_lower]
        else:
            return y

    import h5py
    from scipy import interpolate

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

    ## Interpolate 2 ell_fix to the same scale first ##
    # "Other" side
    if zp is None:
        zp = survey_other['zeropoint']

    x = ell_fix_other['sma'] * survey_other['pixel_scale'] * \
        imutils.phys_size(redshift, is_print=False)
    mu_other = -2.5*np.log10((ell_fix_other['intens'] - skyval[0])/(
        survey_other['pixel_scale'])**2) + zp + filter_corr[0]
    mu_other = fill_defects(mu_other)

    # error: ellipse error + background error
    if 'intens_err' in Table(ell_fix_other).colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'
    ellipse_err = ell_fix_other[intens_err_name]
    ellipse_err[np.isnan(ellipse_err)] = 0.0
    # Quadrature Error, at intensity level
    err_other = np.sqrt(ellipse_err**2 + skystd[0]**2)
    err_other /= ell_fix_other['intens']
    if sum(np.isnan(mu_other)) == 0:
        mask = len(mu_other) - 1
    else:
        mask = np.argwhere(np.isnan(mu_other))[0, 0]
    func1 = interpolate.interp1d(
        x[:mask]**0.25, mu_other[:mask], kind='cubic', fill_value='extrapolate')
    func1_err = interpolate.interp1d(
        x[:mask]**0.25, err_other[:mask], kind='cubic', fill_value='extrapolate')
    x_1 = x[:mask].max()**0.25

    # "Fiducial" side
    x = ell_fix_fid['sma'] * survey_fid['pixel_scale'] * \
        imutils.phys_size(redshift, is_print=False)
    mu_fid = -2.5*np.log10((ell_fix_fid['intens'] - skyval[1])/(
        survey_fid['pixel_scale'])**2) + survey_fid['zeropoint'] + filter_corr[1]
    mu_fid = fill_defects(mu_fid)

    # error: ellipse error + background error
    if 'intens_err' in Table(ell_fix_fid).colnames:
        intens_err_name = 'intens_err'
    else:
        intens_err_name = 'int_err'
    ellipse_err = ell_fix_fid[intens_err_name]
    ellipse_err[np.isnan(ellipse_err)] = 0.0
    # Quadrature Error, at intensity level
    err_fid = np.sqrt(ellipse_err**2 + skystd[1]**2)
    err_fid /= ell_fix_fid['intens']
    if sum(np.isnan(mu_fid)) == 0:
        mask = len(mu_fid) - 1
    else:
        mask = np.argwhere(np.isnan(mu_fid))[0, 0]
    func2 = interpolate.interp1d(
        x[:mask]**0.25, mu_fid[:mask], kind='cubic', fill_value='extrapolate')
    func2_err = interpolate.interp1d(
        x[:mask]**0.25, err_fid[:mask], kind='cubic', fill_value='extrapolate')
    x_2 = x[:mask].max()**0.25

    # Interpolate
    x_input = np.arange(x_min, min(x_1, x_2), 0.05)
    y_other = func1(x_input)
    y_fid = func2(x_input)

    mu_diff = y_fid - y_other
    mu_diff_err = -2.5 / \
        np.log(10) * np.sqrt(func1_err(x_input)**2 + func2_err(x_input)**2)

    if label is not None:
        ax1.plot(x_input, mu_diff, color=linecolor, linewidth=linewidth, linestyle='-',
                 label=r'$\mathrm{' + label + '}$', alpha=1)
        leg = ax1.legend(fontsize=labelsize, loc=labelloc)
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x_input, mu_diff, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
    if plot_err:
        ax1.fill_between(x_input, mu_diff - mu_diff_err, mu_diff +
                         mu_diff_err, color=linecolor, alpha=0.4*alpha, label=None)

    # Set ticks
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
    ylabel = r'$\mu_{\mathrm{DF}} - \mu_i$' + \
        '\n' + '$[\mathrm{mag/arcsec^2}]$'
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)

    # Twin axis with linear scale
    if show_banner is True:
        ax4 = ax1.twiny()
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
        lin_pos = [i**0.25 for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=ticksize)
        ax4.xaxis.set_label_coords(1, 1.035)

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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
        # plt.ylim(ylim)

    # Return
    if ax is None:
        return fig, mu_diff
    else:
        return ax1, mu_diff


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
        y = -2.5 * np.log10((ell_fix['intens'] +
                             offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'] +
                             offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)

    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=alpha*0.3)
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
    ax2.plot(x**0.25, ell_free['ell'], color=linecolor,
             linewidth=linewidth, alpha=alpha)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ylim = ax2.get_ylim()
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0, 0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i, 2))+'$' for i in ytick_pos])
    ax2.set_ylim(ylim)
    if show_hline:
        ax2.axhline(y=mean_e, color=linecolor,
                    alpha=1, linestyle='-.', linewidth=2)

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
        ax3.plot(x**0.25, ell_free['pa_norm'],
                 color=linecolor, linewidth=linewidth, alpha=alpha)
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
            ax3.axhline(y=mean_pa, color=linecolor,
                        alpha=1, linestyle='-.', linewidth=2)

        if physical_unit and show_banner:
            ax4 = ax3.twiny()
            ax4.tick_params(direction='in')
            lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
            lin_pos = [i**0.25 for i in lin_label]
            ax4.set_xticks(lin_pos)
            ax4.set_xlim(ax3.get_xlim())
            ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
            ax4.xaxis.set_label_coords(1, 1.05)
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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
    phys_size = phys_size(redshift, is_print=False)
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
        y = -2.5 * np.log10((ell_fix['intens'] +
                             offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix['int_err']) / (pixel_scale)**2) + zeropoint
        upper_yerr = y_lower - y
        lower_yerr = y - y_upper
        asymmetric_error = [lower_yerr, upper_yerr]
        xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
        ylabel = r'$\mu\,[\mathrm{mag/arcsec^2}]$'
    else:
        x = ell_fix['sma'] * pixel_scale
        y = -2.5 * np.log10((ell_fix['intens'] +
                             offset) / (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10((ell_fix['intens'] + offset +
                      ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
        y_lower = -2.5 * \
            np.log10((ell_fix['intens'] + offset -
                      ell_fix['int_err']) / (pixel_scale) ** 2) + zeropoint
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
        ax1.plot(x**0.25, y, color=linecolor, linewidth=linewidth,
                 linestyle=linestyle, alpha=alpha)

    ax1.fill_between(x**0.25, y_upper, y_lower,
                     color=linecolor, alpha=alpha*0.3)
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
    ax2.plot(x**0.25, ell_free['ell'], color=linecolor,
             linewidth=linewidth, alpha=alpha)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ylim = ax2.get_ylim()
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0, 0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i, 2))+'$' for i in ytick_pos])
    ax2.set_ylim(ylim)
    if show_hline:
        ax2.axhline(y=mean_e, color=linecolor,
                    alpha=1, linestyle='-.', linewidth=2)

    if show_pa:
        # Position Angle profile
        from kungpao import utils

        y = np.abs(ell_free['pa_norm'] -
                   np.nanmean(ell_free['pa_norm'][parange[0]:parange[1]]))

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
            ax3.axhline(y=mean_pa, color=linecolor,
                        alpha=1, linestyle='-.', linewidth=2)

        if physical_unit and show_banner:
            ax4 = ax3.twiny()
            ax4.tick_params(direction='in')
            lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
            lin_pos = [i**0.25 for i in lin_label]
            ax4.set_xticks(lin_pos)
            ax4.set_xlim(ax3.get_xlim())
            ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=30)
            ax4.xaxis.set_label_coords(1, 1.05)
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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
    Plot SBP together, along with median profile (on flux level)

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
    from .profile import skyobj_value
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
        info = str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_free = f[band]['ell_free'].value
        ell_fix = f[band]['ell_fix'].value
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
                show_banner=(k == 0),
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, 60)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))
        f.close()

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        yerr_set = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=np.nanmedian)) for bootarr in y_stack.T])

    y = -2.5*np.log10(np.nanmedian(y_stack, axis=0) /
                      (pixel_scale)**2) + zeropoint
    y_upper = -2.5*np.log10((np.nanmedian(y_stack, axis=0) +
                             yerr_set)/(pixel_scale)**2) + zeropoint
    y_lower = -2.5*np.log10((np.nanmedian(y_stack, axis=0) -
                             yerr_set)/(pixel_scale)**2) + zeropoint
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
        ax1.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
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
    from .profile import skyobj_value
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
        info = str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        img = f[band]['image'].value
        mask = f[band]['mask'].value
        ell_free = f[band]['ell_free'].value
        ell_fix = f[band]['ell_fix'].value

        # Calculate mean ellipticity and pa, which are used for fixed fitting
        #mean_e = info['mean_e']
        #mean_pa = info['mean_pa']

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
            single_label = None  # "S18A\ sky\ objects"
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
                show_banner=(k == 0),
                show_hline=False,
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)

        x_input = np.linspace(x_min, x_max, ninterp)

        # Interpolate for surface brightness
        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens']-off_set, kind='cubic', fill_value='extrapolate')
        if k == 0:
            SB_stack = func(x_input)
        else:
            SB_stack = np.vstack((SB_stack, func(x_input)))

        # Interpolate for ellipticity
        x = ell_free['sma'] * slug.HSC_pixel_scale * \
            phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['ell'])
        func = interpolate.interp1d(
            x[mask]**0.25, ell_free['ell'][mask], kind='cubic', fill_value='extrapolate')
        if k == 0:
            e_stack = func(x_input)
        else:
            e_stack = np.vstack((e_stack, func(x_input)))

        # Interpolate for position angle
        x = ell_free['sma'] * slug.HSC_pixel_scale * \
            phys_size(redshift, is_print=False)
        mask = ~np.isnan(ell_free['pa_norm'])

        func = interpolate.interp1d(x[mask]**0.25,
                                    abs(ell_free['pa_norm']-np.nanmean(
                                        ell_free['pa_norm'][parange[0]:parange[1]]))[mask],
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
        SB_err = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in SB_stack.T])
        e_err = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in e_stack.T])
        pa_err = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in pa_stack.T])

    # ax1: SBP
    y = -2.5*np.log10(np.nanmedian(SB_stack, axis=0) /
                      (pixel_scale)**2) + zeropoint
    y_upper = -2.5 * \
        np.log10((np.nanmedian(SB_stack, axis=0) + SB_err) /
                 (pixel_scale)**2) + zeropoint
    y_lower = -2.5 * \
        np.log10((np.nanmedian(SB_stack, axis=0) - SB_err) /
                 (pixel_scale)**2) + zeropoint
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
        ax1.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
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
    ax2.plot(x_input, y, color=linecolor,
             linewidth=linewidth, linestyle='-', alpha=1)
    ax2.fill_between(x_input, y - e_err, y + e_err, color=fillcolor, alpha=0.4)
    ax2.xaxis.set_major_formatter(NullFormatter())
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax2.set_xlim(x_min, x_max)
    ylim = ax2.get_ylim()
    ax2.set_ylabel(r'$e$', fontsize=35)
    ytick_pos = [0, 0.2, 0.4, 0.6]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels([r'$'+str(round(i, 2))+'$' for i in ytick_pos])
    ax2.set_ylim([0, 0.4])

    if show_pa:
        # ax3: Position angle
        y = np.nanmedian(pa_stack, axis=0)
        ax3.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
        ax3.fill_between(x_input, y - pa_err, y + pa_err,
                         color=fillcolor, alpha=0.4)
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
            SB_err = np.array(
                [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[0].T])
            e_err = np.array(
                [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[1].T])
            pa_err = np.array(
                [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in stack[2].T])

        x_input = stack[3]
        # ax1: SBP
        y = -2.5 * \
            np.log10(np.nanmedian(stack[0], axis=0) /
                     (pixel_scale)**2) + zeropoint
        y_upper = -2.5 * \
            np.log10(
                (np.nanmedian(stack[0], axis=0) + SB_err)/(pixel_scale)**2) + zeropoint
        y_lower = -2.5 * \
            np.log10(
                (np.nanmedian(stack[0], axis=0) - SB_err)/(pixel_scale)**2) + zeropoint
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
            ax1.plot(
                x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
        ax1.fill_between(x_input, y_upper, y_lower,
                         color=fillcolor[k], alpha=0.4)
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
        ax2.plot(
            x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
        ax2.fill_between(x_input, y - e_err, y + e_err,
                         color=fillcolor[k], alpha=0.4)
        ax2.xaxis.set_major_formatter(NullFormatter())
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        ax2.set_xlim(x_min, x_max)
        ylim = ax2.get_ylim()
        ax2.set_ylabel(r'$e$', fontsize=35)
        ytick_pos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ax2.set_yticks(ytick_pos)
        ax2.set_yticklabels([r'$'+str(round(i, 2))+'$' for i in ytick_pos])
        ax2.set_ylim(ylim)

        if show_pa:
            # ax3: Position angle
            y = np.nanmedian(stack[2], axis=0)
            ax3.plot(
                x_input, y, color=linecolor[k], linewidth=linewidth, linestyle=linestyle[k], alpha=1)
            ax3.fill_between(x_input, y - pa_err, y + pa_err,
                             color=fillcolor[k], alpha=0.4)
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
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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
            ax4.set_xticklabels(
                [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=25)
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

# Plot SBP together, and also plot median profile


def mu_diff_stack(obj_cat, band, ax=None,
                  sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0,
                  vertical_line=None, show_banner=True, ismedian=True, linecolor='brown',
                  fillcolor=['#fdcc8a', '#fc8d59', '#d7301f'], linewidth=5, single_alpha=0.3, single_color='firebrick',
                  single_style='-', single_width=1, label=None, ticksize=20, labelsize=20,
                  single_label="S18A\ sky\ objects"):
    import slug
    from slug import imutils
    import h5py
    import pickle
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
        # Load HSC files
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        # Load info
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix_HSC = Table(f[band]['ell_fix'].value)
        f.close()

        if sky_cat is None:
            off_set = 0.0
        else:
            off_set = imutils.skyobj_value(sky_cat,
                                           ra,
                                           dec,
                                           matching_radius=matching_radius,
                                           aperture=aperture,
                                           maxiters=5,
                                           showmedian=False)
        ell_fix_HSC['intens'] -= off_set

        # Load DECaLS files
        with open(obj['decals_dir'].rstrip(' '), 'rb') as f:
            ellipsefit = pickle.load(f)
        # Change the unit of 'intens' to count/pixel
        for filt in ellipsefit['bands']:
            ellipsefit[filt]['intens'] *= (slug.DECaLS_pixel_scale)**2
            ellipsefit[filt]['intens_err'] *= (slug.DECaLS_pixel_scale)**2
        ell_fix_DECaLS = Table(ellipsefit[band[0]])  # r-band ellipse result

        # Label
        if k == 0:
            single_label = single_label
        else:
            single_label = None

        ## Interpolate the two ell_fix to the same "x" first ##
        # HSC side
        x = ell_fix_HSC['sma'] * slug.HSC_pixel_scale * \
            imutils.phys_size(redshift, is_print=False)
        mu_HSC = -2.5 * \
            np.log10((ell_fix_HSC['intens']) /
                     (slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
        mask = np.isnan(mu_HSC)
        func = interpolate.interp1d(x[~mask]**0.25, mu_HSC[~mask],
                                    kind='cubic', fill_value=np.nan, bounds_error=False)
        x_input = np.linspace(x_min, x_max, 60)
        y_HSC = func(x_input)
        y_HSC[x_input > max(x)] = np.nan

        # DECaLS side.
        x = ell_fix_DECaLS['sma'] * slug.DECaLS_pixel_scale * \
            imutils.phys_size(redshift, is_print=False)
        mu_DECaLS = -2.5 * \
            np.log10(
                (ell_fix_DECaLS['intens'])/(slug.DECaLS_pixel_scale)**2) + slug.DECaLS_zeropoint
        mask = np.isnan(mu_DECaLS)
        # func = interpolate.interp1d(x[~mask]**0.25, mu_DECaLS[~mask],
        #                            kind='cubic', fill_value=np.nan, bounds_error=False)
        func = interpolate.interp1d(x[~mask]**0.25, mu_DECaLS[~mask],
                                    kind='cubic', fill_value=np.nan, bounds_error=False)
        x_input = np.linspace(x_min, x_max, 60)
        y_DECaLS = func(x_input)
        y_DECaLS[x_input > max(x)] = np.nan

        mu_diff = y_HSC - y_DECaLS

        if k == 0:
            mu_stack = mu_diff
        else:
            mu_stack = np.vstack((mu_stack, mu_diff))

    with NumpyRNGContext(np.random.randint(10000)):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        yerr_set = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in mu_stack.T])

    y = np.nanmedian(mu_stack, axis=0)
    # Fix the bug that some "y" is derived from mu_stack which only has one non-nan value
    y[np.isnan(yerr_set)] = np.nan
    yerr_set[np.isnan(y)] = np.nan

    # If `nan` at somewhere, interpolate `nan`.
    nanidx = np.where(np.isnan(y))[0]
    if len(nanidx) > 1:
        from sklearn.cluster import KMeans
        X = np.array(list(zip(nanidx, np.zeros_like(nanidx))))
        kmeans = KMeans(n_clusters=2).fit(X)
        labels = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        if (max(centroids[:, 0]) - min(centroids[:, 0]) < 3) and np.ptp(nanidx[labels == 0]) > 2:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value=np.nan)
            y[nanidx[labels == 0]] = func(x[nanidx[labels == 0]]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            yerr_set[nanidx[0]:] = np.nan
            yerr_set[nanidx[0]:] = np.nan
    elif len(nanidx) == 1:
        if nanidx + 1 > len(nanidx) or nanidx - 1 < 0:
            print('Sorry, cannot replace NaN')
        elif abs(y[nanidx - 1] - y[nanidx + 1]) < 0.5:
            print('interpolate NaN')
            from scipy.interpolate import interp1d
            mask = (~np.isnan(y))
            func = interp1d(x[mask]**0.25, y[mask],
                            kind='cubic', fill_value=np.nan)
            y[nanidx] = func(x[nanidx]**0.25)
        else:
            y[nanidx[0]:] = np.nan
            yerr_set[nanidx[0]:] = np.nan
            yerr_set[nanidx[0]:] = np.nan

    if label is not None:
        ax1.plot(x_input, y, color=linecolor, linewidth=linewidth, linestyle='-',
                 label=r'$\mathrm{' + label + '}$', alpha=1)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)

    ax1.fill_between(x_input, y + 1 * yerr_set, y - 1 *
                     yerr_set, color=fillcolor[0], alpha=0.8)
    ax1.fill_between(x_input, y + 2 * yerr_set, y - 2 *
                     yerr_set, color=fillcolor[1], alpha=0.6)
    ax1.fill_between(x_input, y + 3 * yerr_set, y - 3 *
                     yerr_set, color=fillcolor[2], alpha=0.4)

    # Set ticks
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    xlabel = r'$(R/\mathrm{kpc})^{1/4}$'
    ylabel = r'$\mu_{\mathrm{HSC}} - \mu_{\mathrm{DECaLS}}$' + \
        '\n' + '$[\mathrm{mag/arcsec^2}]$'
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)

    # Twin axis with linear scale
    if show_banner is True:
        ax4 = ax1.twiny()
        ax4.tick_params(direction='in')
        lin_label = [1, 2, 5, 10, 50, 100, 150, 300]
        lin_pos = [i**0.25 for i in lin_label]
        ax4.set_xticks(lin_pos)
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_xlabel(r'$\mathrm{kpc}$', fontsize=ticksize)
        ax4.xaxis.set_label_coords(1, 1.025)

        ax4.set_xticklabels(
            [r'$\mathrm{'+str(i)+'}$' for i in lin_label], fontsize=ticksize)
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
        return fig, mu_stack
    return ax1, mu_stack

# Plot SBP together, and also plot median profile


def mu_diff_SB(obj_cat, band, ax=None,
               sky_cat=None, matching_radius=3, aperture='84', x_min=20.0, x_max=32.0,
               vertical_line=None, ismedian=True, linecolor='brown',
               fillcolor=['#fdcc8a', '#fc8d59', '#d7301f'], linewidth=5, single_alpha=0.3, single_color='firebrick',
               single_style='-', single_width=1, label=None, ticksize=20, labelsize=20,
               single_label="S18A\ sky\ objects"):
    import slug
    from slug import imutils
    import h5py
    import pickle
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
        # Load HSC files
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        # Load info
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix_HSC = Table(f[band]['ell_fix'].value)
        f.close()

        if sky_cat is None:
            off_set = 0.0
        else:
            off_set = imutils.skyobj_value(sky_cat,
                                           ra,
                                           dec,
                                           matching_radius=matching_radius,
                                           aperture=aperture,
                                           maxiters=5,
                                           showmedian=False)
        ell_fix_HSC['intens'] -= off_set

        # Load DECaLS files
        ellipsefit = Table.read(obj['decals_dir'])
        ell_fix = Table(
            data=[
                ellipsefit['R_SMA'].data[0],  # pixel
                ellipsefit['R_INTENS'].data[0] * (slug.DECaLS_pixel_scale) **
                2,  # nanomaggie/pixel
                ellipsefit['R_INTENS_ERR'].data[0] * (slug.DECaLS_pixel_scale) **
                2  # nanomaggie/pixel
            ],
            names=['sma', 'intens', 'intens_err'])  # r-band ellipse result

        ell_fix_DECaLS = ell_fix  # r-band ellipse result

        # Label
        if k == 0:
            single_label = single_label
        else:
            single_label = None

        ## Interpolate the two ell_fix to the same "x" first ##
        # HSC side
        x = ell_fix_HSC['sma'] * slug.HSC_pixel_scale * \
            imutils.phys_size(redshift, is_print=False)
        mu_HSC = -2.5 * \
            np.log10((ell_fix_HSC['intens']) /
                     (slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
        mask = np.isnan(mu_HSC)
        func = interpolate.interp1d(x[~mask]**0.25, mu_HSC[~mask],
                                    kind='cubic', fill_value=np.nan, bounds_error=False)
        x_input = np.linspace(1.0, 4.5, 50)
        y_HSC = func(x_input)
        y_HSC[x_input > max(x)] = np.nan

        # DECaLS side.
        x = ell_fix_DECaLS['sma'] * slug.DECaLS_pixel_scale * \
            imutils.phys_size(redshift, is_print=False)
        mu_DECaLS = -2.5 * \
            np.log10(
                (ell_fix_DECaLS['intens'])/(slug.DECaLS_pixel_scale)**2) + slug.DECaLS_zeropoint
        mask = np.isnan(mu_DECaLS)
        func = interpolate.interp1d(x[~mask]**0.25, mu_DECaLS[~mask],
                                    kind='cubic', fill_value=np.nan, bounds_error=False)
        x_input = np.linspace(1.0, 4.5, 50)
        y_DECaLS = func(x_input)
        y_DECaLS[x_input > max(x)] = np.nan
        # Difference
        mu_diff = y_HSC - y_DECaLS

        if k == 0:
            mu_diff_stack = mu_diff
            SB_stack = y_HSC
        else:
            mu_diff_stack = np.vstack((mu_diff_stack, mu_diff))
            SB_stack = np.vstack((SB_stack, y_HSC))

    with NumpyRNGContext(np.random.randint(10000)):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        xerr_set = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in SB_stack.T])
        yerr_set = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in mu_diff_stack.T])

    x = np.nanmedian(SB_stack, axis=0)
    y = np.nanmedian(mu_diff_stack, axis=0)
    # Fix the bug that some "y" is derived from mu_stack which only has one non-nan value
    x[np.isnan(yerr_set)] = np.nan
    y[np.isnan(yerr_set)] = np.nan

    if label is not None:
        ax1.plot(x, y, color=linecolor, linewidth=linewidth, linestyle='-',
                 label=r'$\mathrm{' + label + '}$', alpha=1)
        leg = ax1.legend(fontsize=25, frameon=False, loc='lower left')
        for l in leg.legendHandles:
            l.set_alpha(1)
    else:
        ax1.plot(x, y, color=linecolor, linewidth=linewidth,
                 linestyle='-', alpha=1)

    #ax1.errorbar(x, y, xerr=xerr_set, yerr=yerr_set, color=linecolor, fmt='.', capsize=3)
    ax1.fill_between(x, y + 1 * yerr_set, y - 1 * yerr_set,
                     color=fillcolor[0], alpha=0.8)
    ax1.fill_between(x, y + 2 * yerr_set, y - 2 * yerr_set,
                     color=fillcolor[1], alpha=0.6)
    ax1.fill_between(x, y + 3 * yerr_set, y - 3 * yerr_set,
                     color=fillcolor[2], alpha=0.4)

    # Set ticks
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticksize)
    xlabel = r'$\mu_{\mathrm{HSC}}\,[\mathrm{mag/arcsec^2}]$'
    ylabel = r'$\mu_{\mathrm{HSC}} - \mu_{\mathrm{DECaLS}}$' + \
        '\n' + '$[\mathrm{mag/arcsec^2}]$'
    #ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(xlabel, fontsize=ticksize)
    ax1.set_ylabel(ylabel, fontsize=ticksize)

    # Vertical line
    if vertical_line is not None:
        if len(vertical_line) > 3:
            raise ValueError('Maximum length of vertical_line is 3.')
        ylim = ax1.get_ylim()
        style_list = ['-', '--', '-.']
        for k, pos in enumerate(vertical_line):
            ax1.axvline(x=pos, ymin=0, ymax=1,
                        color='gray', linestyle=style_list[k], linewidth=3, alpha=0.75)
        plt.ylim(ylim)

    # Return
    if ax is None:
        return fig, mu_diff_stack, SB_stack
    return ax1, mu_diff_stack, SB_stack


def SBP_stack_hsc(obj_cat, band, pixel_scale,
                  sky_cat=None, matching_radius=3, aperture='84',
                  x_min=1.0, x_max=4.0, interp_step=0.05):
    """
    Retrieve, interpolate, and save HSC profiles of given catalog. 
    X is in physical unit (kpc)^(1/4).

    Parameters:
        obj_cat: object catalog.
        band: string, such as 'r'.
        pixel_scale: float, pixel scale in arcsec/pixel.
        sky_cat: SkyObject catalog.
        matching_radius: float, in arcmin. We match sky objects around the given object within this radius.
        aperture: string, must be in the `SkyObj_aperture_dic`.
        x_min, x_max: float, in ^{1/4} scale.
        interp_step (float): interpolation step size.

    Returns:
        x_input: in (R**1/4 kpc)
        y_stack: HSC surface brightness profiles, in counts/pixel
    """

    import h5py
    from .imutils import skyobj_value
    from scipy import interpolate

    for k, obj in enumerate(obj_cat):
        if k % 100 == 0:
            print(f'Progress: {k} / {len(obj_cat)}')

        # Load files
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix = Table(f[f'{band}-band']['ell_fix'].value)
        f.close()

        # skyobj
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

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', bounds_error=False, fill_value=np.nan)
        x_input = np.arange(x_min, x_max, interp_step)

        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25 - 0.05] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25 - 0.05] = np.nan
            y_stack = np.vstack((y_stack, temp))

    return x_input, y_stack


def SBP_stack_decals(obj_cat, band, pixel_scale, filt_corr=None, x_min=1.0, x_max=4.0, interp_step=0.05):
    """
    Retrieve, interpolate, and save DECaLS DR9 profiles of given catalog. 
    X is in physical unit (kpc)^(1/4). Update in July 2021.

    Parameters:
        obj_cat: object catalog.
        band: string, such as 'r-band'.
        pixel_scale: float, pixel scale in arcsec/pixel.
        filt_corr (np.array): color correction term between HSC and DECaLS, defined as
            `filt_corr = m_HSC - m_DECaLS`. 
        x_min, x_max: float, in ^{1/4} scale.
        interp_step (float): interpolation step size.

    Returns:
        x_input: in (R**1/4 kpc)
        y_stack: DECaLS surface brightness profiles, counts/pixel
    """
    from scipy import interpolate

    for k, obj in enumerate(obj_cat):
        if k % 100 == 0:
            print(f'Progress: {k} / {len(obj_cat)}')

        # Load files
        ellipsefit = Table.read(obj['decals_dir'])
        ell_fix = Table(
            data=[
                ellipsefit[f'{band.upper()}_SMA'].data[0],  # pixel
                ellipsefit[f'{band.upper()}_INTENS'].data[0] * (slug.DECaLS_pixel_scale) **
                2,  # nanomaggie/pixel
                ellipsefit[f'{band.upper()}_INTENS_ERR'].data[0] * (slug.DECaLS_pixel_scale) **
                2,  # nanomaggie/pixel
                np.ones_like(ellipsefit[f'{band.upper()}_SMA'].data[0]
                             ) * ellipsefit['EPS'].data[0]
            ],
            names=['sma', 'intens', 'intens_err', 'eps'])  # r-band ellipse result
        redshift = obj['z_best']

        # skyobj
        off_set = 0.0

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', bounds_error=False, fill_value=np.nan)
        x_input = np.arange(x_min, x_max, interp_step)

        if filt_corr is not None:
            color_correction = filt_corr[k]
        else:
            color_correction = 0.0

        if k == 0:
            y_stack = func(x_input)
            y_stack *= 10**(-color_correction / 2.5)
            y_stack[x_input > max(x)**0.25 - 0.05] = np.nan
        else:
            temp = func(x_input)
            temp *= 10**(-color_correction / 2.5)
            temp[x_input > max(x)**0.25 - 0.05] = np.nan
            y_stack = np.vstack((y_stack, temp))

    return x_input, y_stack

# Plot SBP together, and also plot median profile


def SBP_stack_new_hsc_magmid(obj_cat, band, pixel_scale, zeropoint, ax=None, physical_unit=False,
                             sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, ninterp=60, show_single=True,
                             vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', linewidth=5,
                             single_alpha=0.3, single_color='firebrick', single_style='-', single_width=1, label=None,
                             single_label="S18A\ sky\ objects"):
    """
    Plot SBP together, along with median profile, but on magnitude level

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
    x_input: corresponding x array.
    """
    import h5py
    from .imutils import skyobj_value
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
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix = Table(f[band]['ell_fix'].value)
        f.close()
        # skyobj
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
                show_banner=(k == 0),
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, ninterp)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))
        f.close()

    x_input = np.linspace(1.0, 4.5, 60)
    y_stack = -2.5 * \
        np.log10(y_stack / (slug.HSC_pixel_scale)**2) + slug.HSC_zeropoint
    y = np.nanmedian(y_stack, axis=0)
    yerr = np.array([np.std(bootstrap(bootarr, 100, bootfunc=np.nanmedian))
                     for bootarr in y_stack.T])

    y_upper = y - yerr
    y_lower = y + yerr
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
        ax1.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
    ax1.fill_between(x_input, y_upper, y_lower, color=fillcolor, alpha=0.4)

    # Return
    if ax is None:
        return fig, y_stack, x_input
    return ax1, y_stack, x_input


def SBP_outskirt_stat_hsc(obj_cat, band, pixel_scale, zeropoint,
                          sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.5, ninterp=60):
    """
    Plot SBP together, along with median profile, but on magnitude level

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
    x_input: corresponding x array.
    """
    import h5py
    import pickle
    from .imutils import skyobj_value
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext

    sma_single_set = []
    SBP_single_set = []
    SBP_single_err_set = []

    for k, obj in enumerate(obj_cat):
        # Load files
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix = Table(f[band]['ell_fix'].value)
        f.close()
        # skyobj
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

        # 1-D profile
        if 'intens_err' in ell_fix.colnames:
            intens_err_name = 'intens_err'
        else:
            intens_err_name = 'int_err'

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, ninterp)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))

        sma_single_set.append(x.data**0.25)
        SBP_single_set.append(3.631 * (ell_fix['intens'].data - off_set) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5))
        # \muJy/arcsec^2
        SBP_single_err_set.append(
            3.631 * (ell_fix[intens_err_name].data) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5))
        # \muJy/arcsec^2
        f.close()

    y_stack = 3.631 * (y_stack) / (pixel_scale)**2 / \
        10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2

    # Return
    return y_stack, x_input, SBP_single_set, SBP_single_err_set, sma_single_set

# Plot SBP together, and also plot median profile


def SBP_outskirt_stat_decals(obj_cat, band, pixel_scale, zeropoint, filt_corr=None,
                             sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.5, ninterp=60):
    """
    Plot SBP together, along with median profile, but on magnitude level

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
    x_input: corresponding x array.
    """

    import h5py
    import pickle
    from .imutils import skyobj_value
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext

    sma_single_set = []
    SBP_single_set = []
    SBP_single_err_set = []

    for k, obj in enumerate(obj_cat):
        # Load files
        with open(obj['decals_dir'].rstrip(' '), 'rb') as f:
            ellipsefit = pickle.load(f)
        # Change the unit of 'intens' to count/pixel
        for filt in ellipsefit['bands']:
            ellipsefit[filt]['intens'] *= (slug.DECaLS_pixel_scale)**2
            ellipsefit[filt]['intens_err'] *= (slug.DECaLS_pixel_scale)**2
        ell_fix = Table(ellipsefit[band[0]])  # r-band ellipse result
        redshift = obj['z_best']
        # skyobj
        off_set = 0.0

        # 1-D profile
        if 'intens_err' in ell_fix.colnames:
            intens_err_name = 'intens_err'
        else:
            intens_err_name = 'int_err'

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, ninterp)

        if filt_corr is not None:
            color_correction = filt_corr[k]
        else:
            color_correction = 0.0

        if k == 0:
            y_stack = func(x_input)
            y_stack *= 10**(-color_correction / 2.5)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp *= 10**(-color_correction / 2.5)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))

        sma_single_set.append(x.data**0.25)
        SBP_single_set.append(3.631 * (ell_fix['intens'].data - off_set) / (
            pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5))
        # \muJy/arcsec^2
        SBP_single_err_set.append(
            3.631 * (ell_fix[intens_err_name].data) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5))
        # \muJy/arcsec^2
        f.close()

    y_stack = 3.631 * (y_stack) / (pixel_scale)**2 / \
        10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2

    # Return
    return y_stack, x_input, SBP_single_set, SBP_single_err_set, sma_single_set


'''
# NO PLOT HERE
def SBP_outskirt_stat_decals(obj_cat, band, pixel_scale, zeropoint,
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.5, ninterp=60):
    """
    Plot SBP together, along with median profile, but on magnitude level
    
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
    x_input: corresponding x array.
    """
    import h5py
    import pickle
    from .imutils import skyobj_value
    from scipy import interpolate
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext

    for k, obj in enumerate(obj_cat):
        # Load files
        with open(obj['decals_dir'].rstrip(' '), 'rb') as f:
            ellipsefit = pickle.load(f)
        # Change the unit of 'intens' to count/pixel
        for filt in ellipsefit['bands']:
            ellipsefit[filt]['intens'] *= (slug.DECaLS_pixel_scale)**2
            ellipsefit[filt]['intens_err'] *= (slug.DECaLS_pixel_scale)**2
        ell_fix = Table(ellipsefit[band[0]]) # r-band ellipse result
        redshift = obj['z_best']
        # skyobj
        off_set = 0.0

        # 1-D profile
        if 'intens_err' in ell_fix.colnames:
            intens_err_name = 'intens_err'
        else:
            intens_err_name = 'int_err'
        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, ninterp)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))
        f.close()
    
    y_stack = 3.631 * (y_stack) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)  #\muJy/arcsec^2  
    # Return
    return y_stack, x_input
    
# Plot SBP together, and also plot median profile
def SBP_outskirt_stat_decals(obj_cat, band, pixel_scale, zeropoint, ax=None, physical_unit=False, 
    sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, ninterp=60, show_single=True, 
    vertical_line=None, ismedian=True, linecolor='brown', fillcolor='orange', linewidth=5,
    single_alpha=0.3, single_color='firebrick', single_style='-', single_width=1, label=None, 
    single_label="S18A\ sky\ objects"):
    """
    Plot SBP together, along with median profile, but on magnitude level
    
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
    x_input: corresponding x array.
    """
    import h5py
    import pickle
    from .imutils import skyobj_value
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
        with open(obj['decals_dir'].rstrip(' '), 'rb') as f:
            ellipsefit = pickle.load(f)
        # Change the unit of 'intens' to count/pixel
        for filt in ellipsefit['bands']:
            ellipsefit[filt]['intens'] *= (slug.DECaLS_pixel_scale)**2
            ellipsefit[filt]['intens_err'] *= (slug.DECaLS_pixel_scale)**2
        ell_fix = Table(ellipsefit[band[0]]) # r-band ellipse result
        redshift = obj['z_best']
        # skyobj
        off_set = 0.0

        if k == 0:
            single_label = single_label
        else:
            single_label = None
        if show_single:
            ax1 = SBP_single_linear(
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

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(1.0, 4.5, ninterp)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))
        f.close()
    
    y_stack = 3.631 * (y_stack) / (pixel_scale)**2 / 10**((zeropoint - 22.5) / 2.5)  #\muJy/arcsec^2  

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        yerr_set = np.array([np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in y_stack.T])
    
    y = np.nanmedian(y_stack, axis=0)
    y_upper = y + yerr_set
    y_lower = y - yerr_set
    
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
        return fig, y_stack, x_input
    return ax1, y_stack, x_input
'''

# Plot linera SBP together, and also plot median profile


def LSBP_stack_new_hsc(obj_cat, band, pixel_scale, zeropoint, ax=None, physical_unit=False,
                       sky_cat=None, matching_radius=3, aperture='84', x_min=1.0, x_max=4.0, ninterp=60, show_single=True,
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
    x_input: corresponding x array.
    """
    import h5py
    from .imutils import skyobj_value
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
        filename = os.path.abspath(os.path.join(
            '/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/IntermediateZ/',
            obj['new_dir']))
        f = h5py.File(filename, 'r')
        info = slug.h5file.str2dic(f['header'].value)
        redshift = info['redshift']
        ra, dec = info['ra'], info['dec']
        ell_fix = Table(f[band]['ell_fix'].value)
        f.close()
        # skyobj
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
                show_banner=(k == 0),
                vertical_line=vertical_line,
                linecolor=single_color,
                linestyle=single_style,
                linewidth=single_width,
                alpha=single_alpha,
                label=single_label)

        x = ell_fix['sma'] * pixel_scale * phys_size(redshift, is_print=False)
        func = interpolate.interp1d(
            x**0.25, ell_fix['intens'] - off_set, kind='cubic', fill_value='extrapolate')
        x_input = np.linspace(x_min, x_max, ninterp)
        if k == 0:
            y_stack = func(x_input)
            y_stack[x_input > max(x)**0.25] = np.nan
        else:
            temp = func(x_input)
            temp[x_input > max(x)**0.25] = np.nan
            y_stack = np.vstack((y_stack, temp))
        f.close()

    with NumpyRNGContext(2333):
        if ismedian:
            btfunc = np.nanmedian
        else:
            btfunc = np.nanmean
        yerr_set = np.array(
            [np.std(bootstrap(bootarr, 100, bootfunc=btfunc)) for bootarr in y_stack.T])

    y = 3.631 * (ell_fix['intens'].data + offset) / (pixel_scale)**2 / \
        10**((zeropoint - 22.5) / 2.5)  # \muJy/arcsec^2

    y = -2.5 * np.log10(np.nanmedian(y_stack, axis=0) /
                        (pixel_scale)**2) + zeropoint
    y_upper = -2.5 * \
        np.log10((np.nanmedian(y_stack, axis=0) + yerr_set) /
                 (pixel_scale)**2) + zeropoint
    y_lower = -2.5 * \
        np.log10((np.nanmedian(y_stack, axis=0) - yerr_set) /
                 (pixel_scale)**2) + zeropoint
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
        ax1.plot(x_input, y, color=linecolor,
                 linewidth=linewidth, linestyle='-', alpha=1)
    ax1.fill_between(x_input, y_upper, y_lower, color=fillcolor, alpha=0.4)

    # Return
    if ax is None:
        return fig, y_stack, x_input
    return ax1, y_stack, x_input
