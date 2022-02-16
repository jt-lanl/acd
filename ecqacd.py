import numpy as np
import scipy.linalg as la

# Convention for image arrays
# [nPixel,nBand]
# [nRow,nSample,nBand]

def imqim(Q,im):
    '''
    Compute x.T * Q * x, for every vector x in im; 
    Assume im is 2d array, with vectors x being rows of im
    '''
    return np.sum( np.dot( im, Q ) * im, axis=1 )

def outprod(v,w=None):
    '''
    given vectors v,w return the outer product: vw'
    if only one vector is given, return outer product with itself: vv'
    '''
    if w is None:
        w = v

    return np.dot(v.reshape(-1,1),w.reshape(1,-1))


def matinv_reg(M,e=1.0e-12):
    d = M.shape[0]
    t = np.trace(M)
    return la.inv(M + e*(t/d)*np.eye(d))

def sqrtm(X):
    U,J,_ = la.svd(X)
    Xsqrt = np.dot( np.dot(U,np.diag(np.sqrt(J))), U.T )
    return Xsqrt

def get_mXYC(imX,imY,mask=None):
    ## Note: mask must an array of booleans
    ## make sure X and Y have same number of pixels
    assert( imX.shape[:-1] == imY.shape[:-1] )
    if mask is not None:
        assert(mask.shape == imX.shape[:-1])
        assert(mask.dtype == np.bool)

    dx = imX.shape[-1]
    dy = imY.shape[-1]

    imX = imX.reshape(-1,dx)
    imY = imY.reshape(-1,dy)

    if mask is not None:
        imX = imX[~mask.ravel(),:]
        imY = imY[~mask.ravel(),:]

    ## Compute mean values
    mX = np.mean(imX,axis=0)
    mY = np.mean(imY,axis=0)

    ## Subtract mean values
    imX = imX - mX.reshape(1,dx)
    imY = imY - mY.reshape(1,dy)

    ## Compute covariance matrices
    nPixels = imX.shape[0]
    X = np.dot(imX.T,imX)/nPixels
    Y = np.dot(imY.T,imY)/nPixels
    C = np.dot(imY.T,imX)/nPixels

    return mX,mY,X,Y,C

def nu_est(zz,d,m=1):
    ''' Given a set of Mahalanobis distances zz = (z-mu)'*R^{-1}*(z-mu)
        Use the moment-method to estimate nu for multivariate-t
    '''
    rnum = np.mean(zz**(1+m/2))
    rden = np.mean(zz**(m/2))
    kappa = rnum/rden
    if kappa <= d + m:
        est_nu = 0
    else:
        est_nu = 2 + m*kappa/(kappa-(d+m))
    return est_nu

def nu_scale(nu,d,zz):
    assert( nu <=0 or nu > 2 )
    if nu <= 0:
        return zz
    else:
        return (nu+d)*np.log(1 + zz/(nu-2))

class cca():
    
    def __init__(self,n_components):
        self.n_components=n_components
        
    def fit(self,imX,imY,mask=None):
        self.dx = imX.shape[-1]
        self.dy = imY.shape[-1]
        self.mX,self.mY,X,Y,C = get_mXYC(imX,imY,mask=mask)

        Xsqrt = la.cholesky(X)
        Xinvsqrt = la.inv(Xsqrt)
        Ysqrt = la.cholesky(Y)
        Yinvsqrt = la.inv(Ysqrt)
        Ctilde = np.dot(np.dot(Yinvsqrt.T,C),Xinvsqrt)

        U,J,Vt = la.svd(Ctilde)
        U = U[:,:self.n_components]
        Vt = Vt[:self.n_components,:]

        self.A = np.dot(Xinvsqrt,Vt.T)
        self.B = np.dot(Yinvsqrt,U)
        return self

    def transform(self,imX,imY):
        ## make sure X and Y are the same size images
        assert( imX.shape[:-1] == imY.shape[:-1] )

        ## and X and Y have same dimension as training images
        assert( imX.shape[-1] == self.dx )
        assert( imY.shape[-1] == self.dy )

        imShape = list(imX.shape); imShape[-1]=-1

        imX = imX.reshape(-1,self.dx)
        imY = imY.reshape(-1,self.dy)

        imX = imX - self.mX.reshape(1,-1)
        imY = imY - self.mY.reshape(1,-1)

        imX = np.dot(imX,self.A)
        imY = np.dot(imY,self.B)

        imX = imX.reshape(imShape)
        imY = imY.reshape(imShape)


        return imX,imY

class acd:
    
    def fit(self,imX,imY,nu=0,mask=None,**kw_xtra):
        self.nBandsX = dx = imX.shape[-1]
        self.nBandsY = dy = imY.shape[-1]

        self.mX,self.mY,X,Y,C = get_mXYC(imX,imY,mask=mask)

        ## Create concatenated matrix ## matlab: [X C'; C Y]
        XCCY = np.vstack( [np.hstack([X, C.T]), 
                           np.hstack([C, Y  ]) ]) 

        ## Invert matrices
        self.Qzz = matinv_reg(XCCY)
        self.Qxx = matinv_reg(X)
        self.Qyy = matinv_reg(Y)

        if nu==-1:
            d = self.nBandsX+self.nBandsY
            imZ = np.vstack( [imX,imY] ).reshape(-1,d) #nb, mean already subtracted
            zz = imqim(self.Qzz,imZ) 
            nu = nu_est(zz,d)
        self.nu = nu

    def fit_init(self,nu=0):
        ## Initializes the incremental fit
        ## Should this just be __init__ ?
        if nu<0:
            raise RuntimeError("Incremental fit cannot accommodate adaptive nu; use nu>=0")
        self.nPixels=0
        self.mX = self.mY = self.X = self.Y = self.C = 0
        self.nBandsX = self.nBandsY = -1
        self.nu = nu

    def fit_update(self,imX,imY,mask=None):

        if self.nPixels == 0:
            ## if this is first update, then define sizes
            self.nBandsX = imX.shape[-1]
            self.nBandsY = imY.shape[-1]
        else:
            ## if not first update, make sure sizes are consistent with first update
            assert( self.nBandsX == imX.shape[-1] )
            assert( self.nBandsY == imY.shape[-1] )

        ## N= number of pixels from previous updates
        ## M= number of pixels in this batch
        N = self.nPixels
        if mask is not None:
            M = np.sum(~mask)
        else:
            M = imX[...,0].size

        ## compute mean and covariances for this batch of pixels
        mX,mY,X,Y,C = get_mXYC(imX,imY,mask=mask)

        ## update covariances
        f = N*M/((N+M)**2)
        self.X = (N*self.X + M*X)/(N+M) + f*outprod(mX-self.mX)
        self.Y = (N*self.Y + M*Y)/(N+M) + f*outprod(mY-self.mY)
        self.C = (N*self.C + M*C)/(N+M) + f*outprod(mY-self.mY, mX-self.mX)

        ## update means
        self.mX = (N*self.mX + M*mX)/(N+M)
        self.mY = (N*self.mY + M*mY)/(N+M)

        ## update count
        self.nPixels = N+M

    def fit_complete(self):
        ## Create concatenated matrix ## matlab: [X C'; C Y]
        XCCY = np.vstack( [np.hstack([self.X, self.C.T]), 
                           np.hstack([self.C, self.Y  ]) ]) 

        ## Invert matrices
        self.Qzz = matinv_reg(XCCY)
        self.Qxx = matinv_reg(self.X)
        self.Qyy = matinv_reg(self.Y)
        

    def get_xi_zxy(self,imX,imY):
        ''' return three Mahalanobis distances: xi_z, xi_y, xi_x
        '''

        imShape = imX.shape[:-1]

        dX = imX.shape[-1]
        dY = imY.shape[-1]

        assert( imX.shape[:-1] == imY.shape[:-1] )
        assert( self.nBandsX == dX )
        assert( self.nBandsY == dY )

        ## Convert to 2d and subtract mean
        imX = imX.reshape(-1,dX) - self.mX.reshape(1,-1)
        imY = imY.reshape(-1,dY) - self.mY.reshape(1,-1)

        ## Concatenate vectors
        imZ = np.hstack( [imX, imY] )

        ## Compute anomalousness (Mahalanobis) at each pixel
        zz = imqim( self.Qzz, imZ )
        xx = imqim( self.Qxx, imX )
        yy = imqim( self.Qyy, imY )

        zz = zz.reshape(imShape)
        xx = xx.reshape(imShape)
        yy = yy.reshape(imShape)

        return zz,xx,yy


    def apply(self,imX,imY,nu=-1,beta_x=1,beta_y=1,**kw_xtra):

        imShape = imX.shape[:-1]

        dX = imX.shape[-1]
        dY = imY.shape[-1]

        assert( imX.shape[:-1] == imY.shape[:-1] )
        assert( self.nBandsX == dX )
        assert( self.nBandsY == dY )

        zz,xx,yy = self.get_xi_zxy(imX,imY)

        ## Estimate nu, if requested (nu==-1) 
        ## and if not already estimated (self.nu==-1)
        if nu == -1:
            nu = self.nu
        if nu == -1:
            self.nu = nu_est(zz,dX+dY)

        ##Compute anomalousness of change
        if (nu == 0):
            ## Gaussian, nu->infinity
            anom = zz - beta_x*xx - beta_y*yy;
        else:
            anom = (nu+dX+dY)*np.log(nu-2+zz) - \
                   beta_x*(nu+dX)*np.log(nu-2+xx) - \
                   beta_y*(nu+dY)*np.log(nu-2+yy);
            #offset is (roughly) expected value 
            offs = (nu+dX+dY)*np.log(nu-2+dX+dY) - \
                   beta_x*(nu+dX)*np.log(nu-2+dX) - \
                   beta_y*(nu+dY)*np.log(nu-2+dY)
            anom -= offs

        anom = anom.reshape(imShape)

        return anom


def echacd(imX,imY,**kwargs):
    '''
    EC-HACD (Elliptically contoured Hyperbolic Anomalous Change Detectoin)
    kwargs include: nu=0,beta_x=1,beta_y=1,mask=None):
    '''

    a = acd()
    a.fit(imX,imY,**kwargs)
    anom = a.apply(imX,imY,**kwargs)

    return anom
    
    
