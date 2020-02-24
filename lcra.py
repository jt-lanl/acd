## LCRA: Local Co-Registration Adjustment
## For anomalous change detection 

from __future__ import division,print_function,absolute_import
import numpy as np

def get_drdc(radius,circular=False):
    '''
    return list of tuples (dr,dc) that define a neighborhood
    with a given radius; either square or circular
    '''
    ## winrange is [0, -1, 1, -2, 2, ... , -radius, radius]
    ## equivalent to range(-radius,radius+1) but ensures '0' goes first
    winrange = [0] + [m*n for n in range(1,1+radius) for m in (-1,1)] 
    drdclist = []
    for dr in winrange:
        for dc in winrange:
            if circular and dr**2 + dc**2 > radius**2:
                continue
            drdclist.append((dr,dc))
    ## consider sorting the list by radius dr**2 + dc**2 ... but why bother?
    #drdclist = sorted( drdclist, key= lambda (r,c): r*r+c*c )
    drdclist = sorted( drdclist, key= lambda rc: rc[0]**2 + rc[1]**2 )
    return drdclist

def lcra_wfcn(imX,imY,acd_fcn,drdclist):

    '''
    LCRA (Local Co-Registration Adjustment) with function acd_fcn
    that maps images X,Y to anomalousness 
    (eg, acd_fcn = lambda X,Y: acd.apply(X,Y,**kwargs))
    drdclist is list of tuples (dr,dc) defining neighborhood
    ** assumes imX and imY are images, ie 3d arrays **
    '''
    nRows, nCols = imShape = imX.shape[:-1]
    assert( imShape == imY.shape[:-1] )
    dX = imX.shape[-1]
    dY = imY.shape[-1]

    anom_min = np.zeros(imShape)+np.inf

    assert( len(drdclist) > 0)
    #drdclist = get_drdc(radius=window,circular=circular)

    for dr,dc in drdclist:
        rsX = slice( max(0,0+dr), min(nRows,nRows+dr) )
        rsY = slice( max(0,0-dr), min(nRows,nRows-dr) )
        csX = slice( max(0,0+dc), min(nCols,nCols+dc) )
        csY = slice( max(0,0-dc), min(nCols,nCols-dc) )

        imXtmp = imX[rsX,csX,:]
        imYtmp = imY[rsY,csY,:]
        anom = acd_fcn(imXtmp,imYtmp)
        #anom_min[rsX,csX] = np.minimum(anom_min[rsX,csX], anom)
        anom_min[rsY,csY] = np.minimum(anom_min[rsY,csY], anom)

    return anom_min     

def slcra_wfcn(imX,imY,fcn,drdclist):
    '''
    SLCRA: Symmetric LCRA
    '''
    anom_XY = lcra_wfcn(imX,imY,fcn,drdclist)
    anom_YX = lcra_wfcn(imY,imX,fcn,drdclist)
    anom = np.maximum(anom_XY,anom_YX)
    return anom

