import numpy as np
import ecqacd
import lcra

def subtiles(npix_rc,n_rc):
    '''Return indices of sub-tiles of a tile,
    if an image img has size (npixr,npixc), 
    return an index tuple irlo,irhi,iclo,ichi
    such that img[irlo:irhi,iclo:ichi] is a sub-tile of img.
    (nr,nc) is the number of rows and columns of sub-tiles
    this function is a generator, so subsequent calls produce new indices
    eg: "for irlo,irhi,iclo,ichi in subtiles(img.shape[:2],(3,3)):"
    loops over 3x3=9 sub-tiles.
    '''
    (npixr,npixc) = npix_rc
    (nr,nc) = n_rc
    ipixr = [i*npixr//nr for i in range(nr+1)]
    ipixc = [i*npixc//nc for i in range(nc+1)]
    for irlo,irhi in zip(ipixr[:-1],ipixr[1:]):
        for iclo,ichi in zip(ipixc[:-1],ipixc[1:]):
            yield irlo,irhi,iclo,ichi

def subtiles_ix(npix_rc,n_rc):
    '''
    Same as subtiles(...), but returns a single index object 
    instead four integers. That is, instead of irlo,irhi,iclo,ichi
    it returns index ix with the property that
    img[ix] is same as img[irlo:irhi,iclo:ichi]
    '''
    (npixr,npixc) = npix_rc
    (nr,nc) = n_rc
    ipixr = [i*npixr//nr for i in range(nr+1)]
    ipixc = [i*npixc//nc for i in range(nc+1)]
    for irlo,irhi in zip(ipixr[:-1],ipixr[1:]):
        for iclo,ichi in zip(ipixc[:-1],ipixc[1:]):
            yield np.ix_(range(irlo,irhi),range(iclo,ichi))

def anomchange(b_img,a_img,nu=0,w=0,mask=None, **kw_xtra):
    '''
    From two input spectral images, return an anomalousness image
    Input:
    b_img, a_img: spectral images have shape [nrows, ncolumns, nbands]
    (number of bands can be different for b_img vs a_img)
    Output:
    acd: anomalopusness image has shape [nrows, ncolumns]
    Keywords:
    nu=0: parameter for multivariate-t (elliptically-contoured) distribution
    w=0: window size for LCRA (misregistration compensation)
    mask=None: boolean image [nrows, ncolumns] of bad pixels 
    Extra keywords:
    beta_x = beta_y = 1: Defines flavor of ACD (chronochrome vs hyperbolic)
    '''
    assert( len(b_img.shape) == 3 )
    assert( len(a_img.shape) == 3 )

    acdobj = ecqacd.acd()
    ecqacd_kwargs = dict(nu=nu, **kw_xtra)
    acdobj.fit(b_img,a_img,nu=nu,mask=mask) 
    acd_fcn = lambda X,Y: acdobj.apply(X,Y,**ecqacd_kwargs)
    if w:
        drdc = lcra.get_drdc(w)
        acd = lcra.lcra_wfcn(b_img,a_img,acd_fcn,drdc)
    else:
        acd = acd_fcn(b_img,a_img)

    if mask is not None:
        ## Ensure that bad pixels are NOT considered anomalous changes
        acd[mask] = np.min(acd)

    return acd

def anomchange_tiled(b_img,a_img,nu=0,w=0,mask=None,tile=None, **kw_xtra):
    '''
    From two input spectral images, return an anomalousness image
    In this tiled version, the image is partitioned into tiles, and for
    each tile, a separate computation is performed.
    Keywords:
    tile=None, if not None, tile should be a tuple of two positive integers,
    indicating the number of rows and columns in the tiling; eg tile=(2,3)
    indicates that the whole image will be partitioned into six tiles, with
    2 rows and 3 columns.
    Further details: 
    See anomchange() above
    '''
    if tile is None:
        return anomchange(b_img,a_img,nu,w,mask,**kw_xtra)

    if len(tile) != 2:
        raise RuntimeError("tile should be a tuple of two values")

    assert( b_img.shape[:-1] == a_img.shape[:-1] )
    nr,nc = b_img.shape[:-1]

    acd = np.zeros((nr,nc),dtype=np.float)

    for irlo,irhi,iclo,ichi in subtiles((nr,nc),tile):
        print("range:",irlo,irhi,iclo,ichi)
        ssix = np.ix_(range(irlo,irhi),range(iclo,ichi))
        imask = None if mask is None else mask[ssix]
        acd[ssix] = anomchange(b_img[ssix],a_img[ssix],nu,w,mask=imask,**kw_xtra)

    return acd
    
def anomchange_tiled_twopass(b_img,a_img,nu=0,w=0,mask=None,tile=None,**kw_xtra):
    '''
    From two input spectral images, return an anomalousness image.
    In this two-pass version, the computation is broken into tiles, but the result
    should be the same as if the whole image had been computed in a single tile.
    Further details: 
    See anomchange_tiled() above
    '''
    if tile is None:
        return anomchange(b_img,a_img,nu,w,mask)

    if len(tile) != 2:
        raise RuntimeError("tile should be a tuple of two values")

    assert( b_img.shape[:-1] == a_img.shape[:-1] )
    nr,nc = b_img.shape[:-1]

    ecqacd_kwargs = dict(w=w, nu=nu, **kw_xtra)

    acdobj = ecqacd.acd()
    acd_fcn = lambda X,Y: acdobj.apply(X,Y,**ecqacd_kwargs)

    ## First pass, use acdobj.fit
    acdobj.fit_init(nu=nu)
    for irlo,irhi,iclo,ichi in subtiles((nr,nc),tile):
        print("1st pass:",irlo,irhi,iclo,ichi)
        ssix = np.ix_(range(irlo,irhi),range(iclo,ichi))
        imask = None if mask is None else mask[ssix]
        acdobj.fit_update( b_img[ssix], a_img[ssix], mask=imask)

    acdobj.fit_complete()

    ## second pass, use acdobj.apply
    acd = np.zeros((nr,nc),dtype=np.float)
    for irlo,irhi,iclo,ichi in subtiles((nr,nc),tile):
        print("2nd pass:",irlo,irhi,iclo,ichi)
        ssix = np.ix_(range(irlo,irhi),range(iclo,ichi))
        acd_fcn = lambda X,Y: acdobj.apply(X,Y,**ecqacd_kwargs)
        if w:
            drdc = lcra.get_drdc(w, circular=True)
            acd[ssix] = lcra.lcra_wfcn(b_img[ssix],a_img[ssix],acd_fcn,drdc)
        else:
            acd[ssix] = acd_fcn(b_img[ssix],a_img[ssix])

    if mask is not None:
        acd[mask] = np.min(acd) ## note, assumes full mask is available, otherwise will need another pass

    return acd
    
