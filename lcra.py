## LCRA: Local Co-Registration Adjustment
## For anomalous change detection 

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
    drdclist = sorted( drdclist, key= lambda rc: rc[0]**2 + rc[1]**2 )
    return drdclist

def lcra_wfcn(imX,imY,acd_fcn,drdclist,reverse=False,symmetric=False):

    '''
    LCRA (Local Co-Registration Adjustment) with function acd_fcn
    that maps images X,Y to anomalousness 
    (eg, acd_fcn = lambda X,Y: acd.apply(X,Y,**kwargs))
    drdclist is list of tuples (dr,dc) defining neighborhood
    ** assumes imX and imY are images, ie 3d arrays **

    reverse==False correspnds to changes X->Y,
    so that anom[r,c] = min_{dr,dc} acd( X[r+dr,c+dc], Y[r,c] )
    (since the anomalous change is in Y at Y[r,c])
    reverse==True corresponds to changes X<-Y, 
    so that anom[r,c] = min_{dr,dc} acd( X[r,c], Y[r+dr,c+dc] )
    (since the anomalous change is at X[r,c])
    symmetric==True corresponds to SLRCR: X<->Y
    so that anom[r,c] = max( anom_XY[r,c], anom_YX[r,c] )
    '''
    nRows, nCols = imShape = imX.shape[:-1]
    assert( imShape == imY.shape[:-1] )
    dX = imX.shape[-1]
    dY = imY.shape[-1]

    anom_min_XY = np.zeros(imShape)+np.inf
    anom_min_YX = np.zeros(imShape)+np.inf

    #drdclist = get_drdc(radius=window,circular=circular)
    assert( len(drdclist) > 0)

    for dr,dc in drdclist:
        rsX = slice( max(0,0+dr), min(nRows,nRows+dr) )
        rsY = slice( max(0,0-dr), min(nRows,nRows-dr) )
        csX = slice( max(0,0+dc), min(nCols,nCols+dc) )
        csY = slice( max(0,0-dc), min(nCols,nCols-dc) )

        imXtmp = imX[rsX,csX,:]
        imYtmp = imY[rsY,csY,:]
        anom = acd_fcn(imXtmp,imYtmp)
        anom_min_YX[rsX,csX] = np.minimum(anom_min_YX[rsX,csX], anom)
        anom_min_XY[rsY,csY] = np.minimum(anom_min_XY[rsY,csY], anom)

    if symmetric:
        return np.maximum(anom_min_XY,anom_min_YX)
    elif reverse:
        return anom_min_YX
    else:
        return anom_min_XY

def slcra_wfcn(imX,imY,fcn,drdclist):
    '''
    SLCRA: Symmetric LCRA
    '''
    return lcra_wfcn(imX,imY,fcn,drdclist,symmetric=True)


