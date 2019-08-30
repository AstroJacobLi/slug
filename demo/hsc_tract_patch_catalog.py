import urllib

for i in range(6):
    URL = 'https://hsc-release.mtk.nao.ac.jp/rsrc/pdr2/koike/survey-area/info/tracts_patches_W-w0{}.txt'.format(i + 1)
    filename = '_tracts_patches_W{}.txt'.format(i + 1)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(URL, filename=filename)
    
    with open(filename, 'r') as f:
        content = f.readlines()
        f.close()
    content = [x.strip('\n') for x in content]
    cat_cen = [x for x in content if 'Tract' in x and 'Patch' in x and 'Center' in x] 
    cat_bound = [x for x in content if 'Tract' in x and 'Patch' in x and 'Corner' in x] 

    ## Centers
    tract_cen = []
    patch_cen = []
    ra_cen = []
    dec_cen = []

    for a in cat_cen:
        tract_cen.append(re.compile("Tract: ([\d]{4})+").search(a).group(1))
        patch_cen.append(re.compile("Patch: ([\d,\d]{3})+").search(a).group(1))
        ra, dec = re.compile("Center \(.*\):\s\((.*?)\s, (.*?)\)").search(a).groups()
        ra_cen.append(float(ra))
        dec_cen.append(float(dec))

    ## Boundaries
    tract_bound = []
    patch_bound = []
    ra_dec_bound = []
    temp = []

    for ind, a in enumerate(cat_bound):
        boundary = re.compile("Corner(\d) \(.*\):\s\((.*?)\s, (.*?)\)").search(a).groups()
        temp.append(boundary)
        if ind % 5 == 4:
            tract_bound.append(re.compile("Tract: ([\d]{4})+").search(a).group(1))
            patch_bound.append(re.compile("Patch: ([\d,\d]{3})+").search(a).group(1))
            ra_dec_bound.append([tuple(map(lambda x: float(x), t[1:])) for t in temp])
            temp = [] 

    ## Make ``fits`` catalog
    survey_cat = Table([
        Column(data=tract_cen, name='tract'),
        Column(data=patch_cen, name='patch'),
        Column(data=ra_cen, name='ra_cen'),
        Column(data=dec_cen, name='dec_cen'),
        Column(data=ra_dec_bound, name='boundary')])
    
    survey_cat.write('HSC_tracts_patches_W{}.fits'.format(i + 1), format='fits', overwrite=True)


from astropy.table import vstack
survey_cat = []
for i in range(6):
    survey_cat.append(Table.read('HSC_tracts_patches_W{}.fits'.format(i + 1)))
big_cat = vstack(survey_cat)
big_cat.write('HSC_tracts_patches_pdr2_wide.fits', format='fits', overwrite=True)