from __future__ import division, print_function, absolute_import

import copy
import warnings
import numpy as np

import astropy

from astropy.table import Table, Column

HSCFILTERS = ['g', 'r', 'i', 'z', 'y']
SNRBANDS = {'g': 80, 'r': 100, 'i': 100, 'z': 100, 'y': 50}

__all__ = ['calc_kcor', 'deredden_hsc_flux_s18a']

def deredden_hsc_flux_s18a(catalog, replace=False, verbose=True):
    """Galactic extinction correction for HSC fluxes.

    This function assumes the columns names from the S18A catalog.
    """
    columns = catalog.colnames

    # Make sure the Galactic extinction values are available.
    assert 'a_g' in columns, "# Cannot find g-band extinction"
    assert 'a_r' in columns, "# Cannot find g-band extinction"
    assert 'a_i' in columns, "# Cannot find g-band extinction"
    assert 'a_z' in columns, "# Cannot find g-band extinction"
    assert 'a_y' in columns, "# Cannot find g-band extinction"

    for col in columns:
        if col[-4:] == 'flux':
            if verbose:
                print("# Correcting flux : %s" % col)

            # HSC use positive values for extinction
            flux_deredden = catalog[col] * 10 ** (0.4 * catalog['a_' + col[0]])

            if replace:
                catalog[col] = flux_deredden
            else:
                catalog.add_column(Column(data=flux_deredden, name=(col + '_dered')))
                catalog.remove_column(col)

    return copy.deepcopy(catalog)


def flux2Mag(flux, zeropoint=27.0):
    """
    Convert HSC flux in unit of ADU to AB magnitude.

    Parameters
    ----------

    flux : scalar or array of float
        Flux of the object in HSC catalog.
    zeropoint : float, optional
        Photometric zeropoint.
        Default : 27.0
    """
    return -2.5 * np.log10(flux) + zeropoint


def mag2Flux(mag, unit='maggy'):
    """
    Convert HSC AB magnitude into physical flux.

    Parameters
    ----------

    mag : scalar or array of float
        Magnitude of the object.
    unit : string, optional
        Unit of the flux. ('maggy/nanomaggy/jy')
        Default : 'maggy'
    """
    flux = 10.0 ** (-0.4 * mag)

    if unit.lower().strip() == 'jy':
        return flux * 3631.0

    if unit.lower().strip() == 'maggy':
        return flux

    if unit.lower().strip() == 'nanomaggy':
        return flux * 1.0E-9

    raise Exception("# Wrong unit! (jy/maggy/nanomaggy)")


def snr2Ivar(flux, snr):
    """
    Estimate the inverse variance given flux and S/N.

    Parameters
    ----------

    flux : scalar or array of float
        Flux of the obejct.
    snr : scalar or array of float
        Signal to noise ratio
    """
    return 1.0 / ((flux / snr) ** 2.0)


def magerr2Ivar(flux, magErr):
    """
    Estimate the inverse variance given flux and magnitude error.
    The reason for this is that we need to correct the magnitude or
    flux for Galactic extinction.

    Parameters
    ----------

    flux : scalar or array of float
        Flux of the obejct.
    magErr : scalar or array of float
        Error of magnitude measurements.
    """
    fluxErr = flux * ((10.0 ** (magErr/2.5)) - 1.0)

    return 1.0 / (fluxErr ** 2.0)


def preISEDFit(inputCat, magType='cmodel_mag',
               filters=HSCFILTERS,
               maggiesCol='cmodel_maggies',
               ivarsCol='cmodel_ivars',
               saveNew=False,
               outputCat=None,
               sortCol=None,
               useSnr=True,
               snrBands=SNRBANDS):
    """
    Convert the magnitude and error into Maggies and IvarMaggies, and
    prepare the catalog for iSEDFit.


    Parameters
    ----------
    inputCat : astropy.table
        Input table object.
    magType : string, optional
        Type of photometry. 'cmodel_mag/parent_mag_convolved_2_2'
        Default : 'mag_cmodel'
    maggiesCol : string, optional
        Name of the new column for maggies.
        Default : 'cmodel_maggies'
    ivarsCol : string, optional
        Name of the new column for ivars
        Default : 'cmodel_ivars'
    saveNew : boolen, optional
        Whether save a new FITS catalog directly.
        Default : False
    outputCat : string, optional
        Name of the output FITS catalog.
        Default : None
    sortCol : string, optional
        Sort the catalog based on specific column.
        Default : None
    useSnr : boolen, optional
        Use S/N to estimate the ivar of flux or not.
        Default : True
    snrBands : dict, optional
        The default S/N in each band.
        Default : SNRBANDS
    """
    cat = copy.deepcopy(inputCat)

    maggies = np.dstack(
        (map(lambda f: mag2Flux(cat[f + magType] - cat['a_' + f],
                                unit='maggy'), filters)))[0]

    if useSnr is False:
        ivars = np.dstack(
            (map(lambda f: magerr2Ivar(
                mag2Flux(cat[f + magType] - cat['a_' + f], unit='maggy'),
                cat[f + magType + '_err']), filters)))[0]
    else:
        ivars = np.dstack(
            (map(lambda f: snr2Ivar(
                mag2Flux(cat[f + magType] - cat['a_' + f], unit='maggy'),
                snrBands[f]), filters)))[0]

    try:
        cat.add_column(Column(name=maggiesCol, data=maggies))
    except Exception:
        cat.remove_column(maggiesCol)
    try:
        cat.add_column(Column(name=ivarsCol, data=ivars))
    except Exception:
        cat.remove_column(ivarsCol)

    if sortCol is not None:
        cat.sort(sortCol)

    if saveNew:
        if outputCat is None:
            newCat = 'maggies_for_isedfit.fits'
        else:
            newCat = outputCat

        cat.write(newCat, format='fits', overwrite=True)

    return cat


def calc_kcor(filter_name, redshift, colour_name, colour_value):
    """
    K-corrections calculator in Python. See http://kcor.sai.msu.ru for the
    reference. Available filter-colour combinations must be present in the
    `coeff` dictionary keys.

    @type   filter_name: string
    @param  filter_name: Name of the filter to calculate K-correction for, e.g.
                         'u', 'g', 'r' for some of the SDSS filters, or 'J2',
                         'H2', 'Ks2' for 2MASS filters (must be present in
                         `coeff` dictionary)
    @type      redshift: float
    @param     redshift: Redshift of a galaxy, should be between 0.0 and 0.5 (no
                         check is made, however)
    @type   colour_name: string
    @param  colour_name: Human name of the colour, e.g. 'u - g', 'g - r',
                         'V - Rc', 'J2 - Ks2' (must be present in `coeff` dictionary)
    @type  colour_value: float
    @param colour_value: Value of the galaxy's colour, specified in colour_name
    @rtype:              float
    @return:             K-correction in specified filter for given redshift and
                         colour
    @version:            2012
    @author:             Chilingarian, I., Melchior. A.-L., and Zolotukhin, I.
    @license:            Simplified BSD license, see http://kcor.sai.msu.ru/license.txt

    Usage example:

        >>> calc_kcor('g', 0.2, 'g - r', 1.1)
        0.5209713975999992
        >>> calc_kcor('Ic', 0.4, 'V - Ic', 2.0)
        0.310069919999993
        >>> calc_kcor('H', 0.5, 'H - K', 0.1)
        -0.14983142499999502

    """
    coeff = {
        'B_BRc': [
            [0,0,0,0],
            [-1.99412,3.45377,0.818214,-0.630543],
            [15.9592,-3.99873,6.44175,0.828667],
            [-101.876,-44.4243,-12.6224,0],
            [299.29,86.789,0,0],
            [-304.526,0,0,0],
        ],

        'B_BIc': [
            [0,0,0,0],
            [2.11655,-5.28948,4.5095,-0.8891],
            [24.0499,-4.76477,-1.55617,1.85361],
            [-121.96,7.73146,-17.1605,0],
            [236.222,76.5863,0,0],
            [-281.824,0,0,0],
        ],

        'H2_H2Ks2': [
            [0,0,0,0],
            [-1.88351,1.19742,10.0062,-18.0133],
            [11.1068,20.6816,-16.6483,139.907],
            [-79.1256,-406.065,-48.6619,-430.432],
            [551.385,1453.82,354.176,473.859],
            [-1728.49,-1785.33,-705.044,0],
            [2027.48,950.465,0,0],
            [-741.198,0,0,0],
        ],

        'H2_J2H2': [
            [0,0,0,0],
            [-4.99539,5.79815,4.19097,-7.36237],
            [70.4664,-202.698,244.798,-65.7179],
            [-142.831,553.379,-1247.8,574.124],
            [-414.164,1206.23,467.602,-799.626],
            [763.857,-2270.69,1845.38,0],
            [-563.812,-1227.82,0,0],
            [1392.67,0,0,0],
        ],

        'Ic_VIc': [
            [0,0,0,0],
            [-7.92467,17.6389,-15.2414,5.12562],
            [15.7555,-1.99263,10.663,-10.8329],
            [-88.0145,-42.9575,46.7401,0],
            [266.377,-67.5785,0,0],
            [-164.217,0,0,0],
        ],

        'J2_J2Ks2': [
            [0,0,0,0],
            [-2.85079,1.7402,0.754404,-0.41967],
            [24.1679,-34.9114,11.6095,0.691538],
            [-32.3501,59.9733,-29.6886,0],
            [-30.2249,43.3261,0,0],
            [-36.8587,0,0,0],
        ],

        'J2_J2H2': [
            [0,0,0,0],
            [-0.905709,-4.17058,11.5452,-7.7345],
            [5.38206,-6.73039,-5.94359,20.5753],
            [-5.99575,32.9624,-72.08,0],
            [-19.9099,92.1681,0,0],
            [-45.7148,0,0,0],
        ],

        'Ks2_J2Ks2': [
            [0,0,0,0],
            [-5.08065,-0.15919,4.15442,-0.794224],
            [62.8862,-61.9293,-2.11406,1.56637],
            [-191.117,212.626,-15.1137,0],
            [116.797,-151.833,0,0],
            [41.4071,0,0,0],
        ],

        'Ks2_H2Ks2': [
            [0,0,0,0],
            [-3.90879,5.05938,10.5434,-10.9614],
            [23.6036,-97.0952,14.0686,28.994],
            [-44.4514,266.242,-108.639,0],
            [-15.8337,-117.61,0,0],
            [28.3737,0,0,0],
        ],

        'Rc_BRc': [
            [0,0,0,0],
            [-2.83216,4.64989,-2.86494,0.90422],
            [4.97464,5.34587,0.408024,-2.47204],
            [-57.3361,-30.3302,18.4741,0],
            [224.219,-19.3575,0,0],
            [-194.829,0,0,0],
        ],

        'Rc_VRc': [
            [0,0,0,0],
            [-3.39312,16.7423,-29.0396,25.7662],
            [5.88415,6.02901,-5.07557,-66.1624],
            [-50.654,-13.1229,188.091,0],
            [131.682,-191.427,0,0],
            [-36.9821,0,0,0],
        ],

        'U_URc': [
            [0,0,0,0],
            [2.84791,2.31564,-0.411492,-0.0362256],
            [-18.8238,13.2852,6.74212,-2.16222],
            [-307.885,-124.303,-9.92117,12.7453],
            [3040.57,428.811,-124.492,-14.3232],
            [-10677.7,-39.2842,197.445,0],
            [16022.4,-641.309,0,0],
            [-8586.18,0,0,0],
        ],

        'V_VIc': [
            [0,0,0,0],
            [-1.37734,-1.3982,4.76093,-1.59598],
            [19.0533,-17.9194,8.32856,0.622176],
            [-86.9899,-13.6809,-9.25747,0],
            [305.09,39.4246,0,0],
            [-324.357,0,0,0],
        ],

        'V_VRc': [
            [0,0,0,0],
            [-2.21628,8.32648,-7.8023,9.53426],
            [13.136,-1.18745,3.66083,-41.3694],
            [-117.152,-28.1502,116.992,0],
            [365.049,-93.68,0,0],
            [-298.582,0,0,0],
        ],

        'FUV_FUVNUV': [
            [0,0,0,0],
            [-0.866758,0.2405,0.155007,0.0807314],
            [-1.17598,6.90712,3.72288,-4.25468],
            [135.006,-56.4344,-1.19312,25.8617],
            [-1294.67,245.759,-84.6163,-40.8712],
            [4992.29,-477.139,174.281,0],
            [-8606.6,316.571,0,0],
            [5504.2,0,0,0],
        ],

        'FUV_FUVu': [
            [0,0,0,0],
            [-1.67589,0.447786,0.369919,-0.0954247],
            [2.10419,6.49129,-2.54751,0.177888],
            [15.6521,-32.2339,4.4459,0],
            [-48.3912,37.1325,0,0],
            [37.0269,0,0,0],
        ],

        'g_gi': [
            [0,0,0,0],
            [1.59269,-2.97991,7.31089,-3.46913],
            [-27.5631,-9.89034,15.4693,6.53131],
            [161.969,-76.171,-56.1923,0],
            [-204.457,217.977,0,0],
            [-50.6269,0,0,0],
        ],

        'g_gz': [
            [0,0,0,0],
            [2.37454,-4.39943,7.29383,-2.90691],
            [-28.7217,-20.7783,18.3055,5.04468],
            [220.097,-81.883,-55.8349,0],
            [-290.86,253.677,0,0],
            [-73.5316,0,0,0],
        ],

        'g_gr': [
            [0,0,0,0],
            [-2.45204,4.10188,10.5258,-13.5889],
            [56.7969,-140.913,144.572,57.2155],
            [-466.949,222.789,-917.46,-78.0591],
            [2906.77,1500.8,1689.97,30.889],
            [-10453.7,-4419.56,-1011.01,0],
            [17568,3236.68,0,0],
            [-10820.7,0,0,0],
        ],

        'H_JH': [
            [0,0,0,0],
            [-1.6196,3.55254,1.01414,-1.88023],
            [38.4753,-8.9772,-139.021,15.4588],
            [-417.861,89.1454,808.928,-18.9682],
            [2127.81,-405.755,-1710.95,-14.4226],
            [-5719,731.135,1284.35,0],
            [7813.57,-500.95,0,0],
            [-4248.19,0,0,0],
        ],

        'H_HK': [
            [0,0,0,0],
            [0.812404,7.74956,1.43107,-10.3853],
            [-23.6812,-235.584,-147.582,188.064],
            [283.702,2065.89,721.859,-713.536],
            [-1697.78,-7454.39,-1100.02,753.04],
            [5076.66,11997.5,460.328,0],
            [-7352.86,-7166.83,0,0],
            [4125.88,0,0,0],
        ],

        'i_gi': [
            [0,0,0,0],
            [-2.21853,3.94007,0.678402,-1.24751],
            [-15.7929,-19.3587,15.0137,2.27779],
            [118.791,-40.0709,-30.6727,0],
            [-134.571,125.799,0,0],
            [-55.4483,0,0,0],
        ],

        'i_ui': [
            [0,0,0,0],
            [-3.91949,3.20431,-0.431124,-0.000912813],
            [-14.776,-6.56405,1.15975,0.0429679],
            [135.273,-1.30583,-1.81687,0],
            [-264.69,15.2846,0,0],
            [142.624,0,0,0],
        ],

        'J_JH': [
            [0,0,0,0],
            [0.129195,1.57243,-2.79362,-0.177462],
            [-15.9071,-2.22557,-12.3799,-2.14159],
            [89.1236,65.4377,36.9197,0],
            [-209.27,-123.252,0,0],
            [180.138,0,0,0],
        ],

        'J_JK': [
            [0,0,0,0],
            [0.0772766,2.17962,-4.23473,-0.175053],
            [-13.9606,-19.998,22.5939,-3.99985],
            [97.1195,90.4465,-21.6729,0],
            [-283.153,-106.138,0,0],
            [272.291,0,0,0],
        ],

        'K_HK': [
            [0,0,0,0],
            [-2.83918,-2.60467,-8.80285,-1.62272],
            [14.0271,17.5133,42.3171,4.8453],
            [-77.5591,-28.7242,-54.0153,0],
            [186.489,10.6493,0,0],
            [-146.186,0,0,0],
        ],

        'K_JK': [
            [0,0,0,0],
            [-2.58706,1.27843,-5.17966,2.08137],
            [9.63191,-4.8383,19.1588,-5.97411],
            [-55.0642,13.0179,-14.3262,0],
            [131.866,-13.6557,0,0],
            [-101.445,0,0,0],
        ],

        'NUV_NUVr': [
            [0,0,0,0],
            [2.2112,-1.2776,0.219084,0.0181984],
            [-25.0673,5.02341,-0.759049,-0.0652431],
            [115.613,-5.18613,1.78492,0],
            [-278.442,-5.48893,0,0],
            [261.478,0,0,0],
        ],

        'NUV_NUVg': [
            [0,0,0,0],
            [2.60443,-2.04106,0.52215,0.00028771],
            [-24.6891,5.70907,-0.552946,-0.131456],
            [95.908,-0.524918,1.28406,0],
            [-208.296,-10.2545,0,0],
            [186.442,0,0,0],
        ],

        'r_gr': [
            [0,0,0,0],
            [1.83285,-2.71446,4.97336,-3.66864],
            [-19.7595,10.5033,18.8196,6.07785],
            [33.6059,-120.713,-49.299,0],
            [144.371,216.453,0,0],
            [-295.39,0,0,0],
        ],

        'r_ur': [
            [0,0,0,0],
            [3.03458,-1.50775,0.576228,-0.0754155],
            [-47.8362,19.0053,-3.15116,0.286009],
            [154.986,-35.6633,1.09562,0],
            [-188.094,28.1876,0,0],
            [68.9867,0,0,0],
        ],

        'u_ur': [
            [0,0,0,0],
            [10.3686,-6.12658,2.58748,-0.299322],
            [-138.069,45.0511,-10.8074,0.95854],
            [540.494,-43.7644,3.84259,0],
            [-1005.28,10.9763,0,0],
            [710.482,0,0,0],
        ],

        'u_ui': [
            [0,0,0,0],
            [11.0679,-6.43368,2.4874,-0.276358],
            [-134.36,36.0764,-8.06881,0.788515],
            [528.447,-26.7358,0.324884,0],
            [-1023.1,13.8118,0,0],
            [721.096,0,0,0],
        ],

        'u_uz': [
            [0,0,0,0],
            [11.9853,-6.71644,2.31366,-0.234388],
            [-137.024,35.7475,-7.48653,0.655665],
            [519.365,-20.9797,0.670477,0],
            [-1028.36,2.79717,0,0],
            [767.552,0,0,0],
        ],

        'Y_YH': [
            [0,0,0,0],
            [-2.81404,10.7397,-0.869515,-11.7591],
            [10.0424,-58.4924,49.2106,23.6013],
            [-0.311944,84.2151,-100.625,0],
            [-45.306,3.77161,0,0],
            [41.1134,0,0,0],
        ],

        'Y_YK': [
            [0,0,0,0],
            [-0.516651,6.86141,-9.80894,-0.410825],
            [-3.90566,-4.42593,51.4649,-2.86695],
            [-5.38413,-68.218,-50.5315,0],
            [57.4445,97.2834,0,0],
            [-64.6172,0,0,0],
        ],

        'z_gz': [
            [0,0,0,0],
            [0.30146,-0.623614,1.40008,-0.534053],
            [-10.9584,-4.515,2.17456,0.913877],
            [66.0541,4.18323,-8.42098,0],
            [-169.494,14.5628,0,0],
            [144.021,0,0,0],
        ],

        'z_rz': [
            [0,0,0,0],
            [0.669031,-3.08016,9.87081,-7.07135],
            [-18.6165,8.24314,-14.2716,13.8663],
            [94.1113,11.2971,-11.9588,0],
            [-225.428,-17.8509,0,0],
            [197.505,0,0,0],
        ],

        'z_uz': [
            [0,0,0,0],
            [0.623441,-0.293199,0.16293,-0.0134639],
            [-21.567,5.93194,-1.41235,0.0714143],
            [82.8481,-0.245694,0.849976,0],
            [-185.812,-7.9729,0,0],
            [168.691,0,0,0],
        ],

    }

    c = coeff[filter_name + '_' + colour_name.replace(' - ', '')]
    kcor = 0.0

    for x, a in enumerate(c):
	    for y, b in enumerate(c[x]):
		    kcor += c[x][y] * redshift**x * colour_value**y

    return kcor
