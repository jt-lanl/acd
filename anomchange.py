'''
Anomalous Change Detection
'''
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

import acdtiled

def commandline():
    argparser = argparse.ArgumentParser()
    paa = argparser.add_argument
    paa("-w",type=int,default=0,help="Window size for LCRA")
    paa("--beta","-b",default=(1,1),nargs=2,type=float,help="beta_x and beta_y")
    paa("--nu",type=float,default=0,help="EC nu parameter: 0 or 2+")
    paa("--tile",type=int,nargs=2,help="Pair of integers, eg '5 5' for 5x5 tiles")
    paa("--twopass",action="store_true",help="Use two-pass algorithm")
    paa("--imagefile","-i",required=True,
        help="HDF5 file with items imgA, imgB, maskAB, maskBA")
    paa("--show",action="store_true",help="show ACD image in a pop-up window")
    paa("--acdclip",type=float,help="Clip the ACD image")
    paa("--mask",action="store_true",help="hack a mask into the mix")
    args = argparser.parse_args()
    return args

def main(args):

    with h5py.File(args.imagefile,'r') as f:
        
        fimA = np.array( f['imgA'] )
        fimB = np.array( f['imgB'] )
        fmask = None

    ## hack a mask
    if args.mask:
        fmask = np.zeros( fimA.shape[:-1], dtype=np.bool )
        nr,nc = fimA.shape[:-1]
        fmask[nr//3:nr//2,nc//3:nc//2] = True

    ## we can test the case of different number of bands in each image
    #fimA = fimA[...,:5]
    #fimB = fimB[...,:4]

    print("shapes:",fimA.shape,fimB.shape)


    bx,by = args.beta

    acd_algorithm = acdtiled.anomchange_tiled_twopass if args.twopass else acdtiled.anomchange_tiled

    acd = acd_algorithm(fimB,fimA,nu=args.nu,w=args.w,beta_x=bx,beta_y=by,mask=fmask,tile=args.tile)

    print("acd",acd.shape,np.min(acd),np.max(acd))

    if args.show:
        if fmask is not None:
            plt.imshow(fmask,cmap="gray")
            plt.figure()

        if args.acdclip:
            plt.imshow(np.clip(acd,0,args.acdclip),cmap='gray_r')
        else:
            plt.imshow(acd,cmap='viridis_r')
        plt.colorbar()
        plt.show()
    
if __name__ == "__main__":
    args = commandline()
    main(args)

    
