# BACKGROUND

ACD (Anomalous Change Detection) is a collection of python routines
that implement algorithms for detecting anomalous changes in pairs of
images.  The images can be panchromatic or multispectral (or
hyperspectral), and for most of the algorithms, the two images need
not have the same number of spectral bands. The general idea is to
distinguish actual changes, which are assumed to be rare, from
pervasive differences.  Most of the algorithms in this package are
described in (or are variants of the algorithms described in) [1].  A
particular variant, the Elliptically-Contoured Hyperbolic Anomalous
Change Detector (EC-HACD), is described in a later publication [2].
The code also incorporates an adaptive scheme to ameliorate residual
misregistration between the images, based on the approach described in
[3].  Although a large class of these algorithms (those that are
distribution-based, rather than difference-based) have been patented
[US 7,953,280 B2 (May 31,2011)], that patent has since been abandoned.

With the focus on generic "anomalous" changes (ie, pixels for which
changes are notably unlike the changes that occur throughout the image
pair), as opposed to specific targeted changes, there is no single
application space that is targeted. But potentially useful
applications include surveillance, situational awareness, and broad
area search.

ACD was developed at Los Alamos National Laboratory under the
Laboratory Directed Research and Development (LDRD) program.  This
software was originally developed in Matlab, as part of an internally
funded project (LDRD20080040DR, FY 2008-2010) called "Automated Change
Detection for Remote Sensing Imagery (ACDRSI)."  Further development
occurred under the auspices of a second internally funded project
(LDRD20180529ECR, FY 2018-2019) on "Geospatial Change Surveillance
with Heterogeneous Data."  This more recent development included the
re-implementation into the python language, as well as further
utilities for breaking large images into tiles, and for masking out
bad pixels.


# DETAILS

This code includes both the Gaussian and EC versions of: the basic
hyperbolic anomalous change detection (HACD) algorithm, along with
chronochrome (CC), and in fact any variant of ACD that can be
expressed in the form xi(z)-bx\*xi(x)-by\*xi(y), where xi(.) is defined
in [2]. For instance, bx=by=1 corresponds to HACD, while bx=1,by=0 is
one of the chronochrome variants.

This package also incorporates LCRA (local co-registration
adjustment), as defined in [3]. And it permits the use of a bad-pixel
mask.

Code is also provided for two kinds of tiling: one-pass and
two-pass. In one-pass tiling, ACD is performed on each tile
separately.  For two-pass tiling, the first pass collects statistics
over the whole image, and based on those statistics, creates an
anomalous change detector.  For the second pass, that detector is
applied to each tile.

As currently implemented, the tiling code requires that the full image
be read into a single numpy array.  This requirement kind of defeats
the purpose of tiling, but the idea (for now) is to provide a
template.  Once a scheme for reading individual tiles from a file is
identified, it can be incorporated into the template.  This scheme
should probably be abstracted so that one can separate the file
reading from the tile processing, but I didn't want to do that
abstraction until I had a good concrete example of how that might work
for a real file.  The current code uses HDF5, but a good argument can
be made for GeoTIFF.

## In this package

* ecqacd.py: core implementation of basic ACD algorithms
* lcra.py: provides utilities for implements LCRA algorithm
* acdtiled.py: combines ecqacd and lcra along with tiling
* anomchange.py: main code that runs ACD with specified options
  on a specified input file; optionally puts image to screen

**Although 'anomchange.py' can be used directly, it is imagined that in practice it will be used as template or example code; most users will want to write their own code, and that that code will import functions from 'ecqacd.py' and other libraries.**

Also included in this package is a sample image pair dataset based on the Viareggio trial [4,5]:
* viareg.h5

This is an HDF5 file, and if you have the 'h5ls' utility you can see that two objects are imcluded in it

    $h5ls viareg.h5    
    imgA                     Dataset {375, 450, 127}   
    imgB                     Dataset {375, 450, 127}   

Here imgA and imgB are the two 127-band images.

## Usage of 'anomchange.py'

An example usage would be to run on the command line:

    $python anomchange.py -i viareg.h5

The output is the minimum and maximum value of the acd (anomalousness) array.  That's not very practical, but it's handy for seeing how that range changes (or, in some cases, hopefully doesn't change) as you change various tiling options.  The whole array is created in anomcahnge.py, so you can mine that for whatever information you need.

There are several command line options, and you can run 

    $python anomchange.py -h

to get a description of all of them.

For instance, you can run

    $python anomchange.py -i viareg.h5 --show

and it should pop up a window with an image of the anomalousness array.  If you do use --show, it is handy to clip the values in the array to make anomalousness image more interpretable; eg '--acdclip 250' for this dataset shows a lot of structure.

### LCRA

Local co-registration adjustment (LCRA) [3] suppresses a lot of small isolated anomalies, especially small linear features, that are caused by small misregistration effects; eg compare

    $python anomchange.py -i viareg.h5 --show --acdclip 250 -w 0  
    $python anomchange.py -i viareg.h5 --show --acdclip 250 -w 1
    $python anomchange.py -i viareg.h5 --show --acdclip 250 -w 2  

It really shows off the effect of w>0 (which is the LCRA radius).

### Tiling

For tiling, you can run:

    $python anomchange.py -i viareg.h5 --tile 2 2  // runs one-pass ACD on each of 4 tiles   
    $python anomchange.py -i viareg.h5 --tile 2 2 --twopass  // runs two-pass ACD on each of 4 tiles   
    $python anomchange.py -i viareg.h5 --tile 1 1 --twopass  // runs two-pass ACD but on the whole image (useful for debugging)   

In two-pass mode, you should see that the range of anomalousness is the same, independent of the number of tiles.

### EC

You can also get EC (elliptically-contoured) algorithms [2] by using nu>2; eg

    $python anomchange.py -i viareg.h5 --nu 10

## Other ACD algorithms

Or you can use chronochrome instead of the default HACD:

    $python anomchange.py -i viareg.h5 --beta 0 1

Or you can do straight anomaly detection on the stacked images:

    $python anomchange.py -i viareg.h5 --beta 0 0

### Mask option

A kind of a hack is the "--mask" option, which creates a mask, a box just up-left of the center; the idea is to have something to test the mask code. In practice of course, the mask would be supplied as a separate image.

    $python anomchange.py -i viareg.h5 --show --mask

For a pixel under mask (ie, a pixel i,j for which mask[i,j]==True), that pixel will not contribute to the estimation of covariance matrices, and the ACD value at that pixel will be set to min(acd); ie, equal to the least anomalous pixel.  A case could be made for setting it to zero or some other fixed value instead.  Another case could be made for going ahead and computing its anomalousness value (ie, mask it out only for the purpose of the covariance matrix computation). But the purpose of this exercise is just to show how mask works; when you know what requirements you have (or what experiments you want do) with masks, you can code accordingly.

## Reminder about 'anomchange.py'

In general, the routine anomchange.py is something that you would, in any operational scenario, replace wholesale with your own interface to your own data.

# REFERENCES

[1] J. Theiler. Quantitative comparison of quadratic covariance-based
anomalous change detectors. Applied Optics 47 (2008) F12-F26.

[2] J. Theiler, C. Scovel, B. Wohlberg, and B. R. Foy. Elliptically
contoured distributions for anomalous change detection in
hyperspectral imagery. IEEE Geoscience and Remote Sensing Letters 7
(2010) 271-275.

[3] J. Theiler and B. Wohlberg. Local Co-Registration Adjustment for
Anomalous Change Detection. IEEE Trans. Geoscience and Remote Sensing
50 (2012) 3107-3116.

[4] N. Acito, S. Matteoli, A. Rossi, M. Diani, and G. Corsini.
Hyperspectral airborne 'Viareggio 2013 Trial' data collection for
detection algorithm assessment. IEEE J. Selected Topics in Applied
Earth Observations and Remote Sensing 9 (2016) 2365-2376.

[5] J. Theiler, M. Kucer, and A. Ziemann. Experiments in Anomalous
Change Detection with the Viareggio 2013 Trial Dataset. Proc. SPIE
11392 (2020) 1139211.

# LANL C Number

I was told: "Be sure to state your LANL C number (C20118 for ACD) in
your README.md or other conspicuous place."

