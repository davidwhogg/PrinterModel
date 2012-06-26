'''
This file is part of PrinterModel.
Copyright 2012 David W. Hogg (NYU) <http://cosmo.nyu.edu/hogg/>.
'''

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':8})
    rc('text', usetex=True)
import numpy as np
import scipy.optimize as op
import Image as im

def logistic(x):
    '''
    Compute and return the logistic function of x.  x can be anything of which you
    can take the np.exp(x).  The return value q will have 0<q<1.
    '''
    return 1. / (1. + np.exp(-x))

def inverse_logistic(q):
    '''
    Compute and return the number x for which the logistic function
    gives the input q.  The input can be any number 0<q<1 of which you
    can take the np.log().
    '''
    return -1. * np.log(1. / q - 1.)

class hoggprinter():
    '''
    A class representing the physical CMYK printer model.

    Initialize with three parameters: The first (delta_K) sets the
    blackness of the black ink.  The second (delta_d) sets the
    absorption of the CMY inks of the "diagnoal" (appropriate RGB
    color) light.  The third (delta_o) sets the absorption of the CMY
    inks of the "off-diagonal" (inappropriate RGB color) light.

    Note: Internal hard-set magic number self.epsilon; it sets the
    printer's preference to use black ink.  This was set by a process
    of trial and error, monitoring conversions of hard colors.
    '''
    epsilon = 0.01 # magic number

    def __init__(self, delta_K, delta_d, delta_o):
        self.eta = np.array(
            [[1. - delta_d, delta_o, delta_o, 1. - delta_K],
             [delta_o, 1. - delta_d, delta_o, 1. - delta_K],
             [delta_o, delta_o, 1. - delta_d, 1. - delta_K]])
        # a cache for prior conversions
        self.cache = {}
        self.conversions = 0
        print 'created', self
        return None

    def __str__(self):
        s = 'hoggprinter instance'
        s += '\n  epsilon = %f' % self.epsilon
        s += '\n  eta = %s' % self.eta
        s += '\n  cache contains %d conversions' % len(self.cache)
        return s

    def cmyk2rgb(self, cmyk):
        '''
        Input: 4-element ndarray of CMYK values on [0, 1] for a single
        image pixel.

        Output: 3-element ndarray of RGB values on [0, 1] for a single
        pixel.
        '''
        return np.prod(1. - self.eta * cmyk, axis=1)

    def rgb2cmyk_stoopid(self, rgb):
        '''
        Input: 3-element ndarray of RGB values on [0, 1] for a single
        pixel.

        Output: 4-element ndarray of CMYK values on [0, 1] for a single
        image pixel.

        This bad (wrong) algorithm is just used to initialize the
        correct (non-stoopid) optimization algorithm.
        '''
        return np.append(1. - rgb, 1. - np.max(rgb))

    def rgb2cmyk(self, rgb):
        '''
        Input: 3-element ndarray of RGB values on (0, 1) for a single
        pixel.

        Output: 4-element ndarray of CMYK values on [0, 1] for a single
        image pixel.

        Internally, this code works by optimizing a chi-squared-like
        objective function based on the forward function cmyk2rgb().
        The objective function weakly prefers using K ink with the
        self.epsilon parameter.

        Note: The code contains a 0.998: Why?  It is for a good
        reason.

        Note: This code throws warnings on RGB values equal to 0. or
        1. because of the logistic function.  Think about it!
        '''
        def chi(pars):
            cmyk = logistic(pars)
            return np.append(rgb - self.cmyk2rgb(cmyk), self.epsilon * (0.998 - cmyk[3]))
        pars = inverse_logistic(self.rgb2cmyk_stoopid(rgb))
        bestpars, status = op.leastsq(chi, pars, maxfev=1000)
        if (status not in [1, 2, 3, 4]):
            print 'rgb2cmyk issue:', status, rgb, self.float2byte(rgb)
            print self.cmyk2rgb(logistic(bestpars)), self.float2byte(self.cmyk2rgb(logistic(bestpars)))
            print logistic(bestpars), self.float2byte(logistic(bestpars))
            if (self.float2byte(rgb) == self.float2byte(self.cmyk2rgb(logistic(bestpars)))):
                print '(no need to panic)'
            else:
                print 'PANIC PANIC PANIC PANIC!'
            print
        return logistic(bestpars)

    def byte2float(self, b):
        '''
        Input: byte tuple on [0, 255].  No input checking; use at your
        own risk!

        Output: float ndarray on [0, 1].

        Note that f = (b + 0.5) / 256.  Think about it!
        '''
        return (np.array(b).astype(float) + 0.5) / 256.

    def float2byte(self, f):
        '''
        Input: float array on [0, 1].  No input checking; use at your
        own risk!

        Output: byte tuple on [0, 255].

        Note at b = floor(f * 256.).  Think about it!
        '''
        return tuple(np.clip((np.array(f) * 256.).astype(int), 0, 255))

    def rgb2cmyk_bytes(self, rgb_bytes):
        '''
        Input: 3-element tuple of RGB byte values on [0, 255] for a
        single pixel.

        Output: 4-element tuple of CMYK byte values on [0, 255] for a
        single image pixel.

        Note that floats on [0, 1] and bytes on [0, 255] are related
        via self.float2byte() and self.byte2float().  Those functions
        also do some ndarray and tuple-ification.
        '''
        self.conversions += 1
        if (self.conversions % 1024) == 0:
            print self.conversions, self
        try:
            cmyk_bytes = self.cache[rgb_bytes]
        except KeyError:
            cmyk_bytes = self.float2byte(self.rgb2cmyk(self.byte2float(rgb_bytes)))
            self.cache[rgb_bytes] = cmyk_bytes
        return cmyk_bytes

    def rgb2cmyk_image(self, ifd):
        '''
        Input: File descriptor (produced by im.open()) for an input
        RGB image.

        Output: File descriptor for an output CMYK image.
        '''
        ofd2 = im.new('CMYK', ifd.size)
        odata2 = [self.rgb2cmyk_bytes(d) for d in ifd.getdata()]
        ofd2.putdata(odata2)
        return ofd2

    def rgb2cmyk_image_file(self, ifn, ofn):
        '''
        Input: File names, one for the input RGB image on disk (can be
        an RGBa image too, but we will flatten / destroy the alpha
        channel), and one for the destination for the CMYK file on
        disk.

        Output:  Nothing (but the output file is written).

        Internally runs self.rgb2cmyk_image()
        '''
        ifd = im.open(ifn, mode='r').convert('RGB') # this gets rid of alpha channel!
        ofd2 = self.rgb2cmyk_image(ifd)
        ofd2.save(ofn)
        return None

    def test(self, rgb, verbose=True):
        '''
        A test function that assesses the quality of the CMYK
        representation of a given set of RGB values for one pixel.
        '''
        cmyk3 = self.rgb2cmyk(rgb)
        rgb4 = self.cmyk2rgb(cmyk3)
        cmyk5 = self.rgb2cmyk(rgb4)
        reperr = np.max(np.abs(rgb4 - rgb))
        if verbose:
            print 'rgb input:', rgb
            print 'cmyk conversion:', cmyk3
            print 'conversion error:', np.max(np.abs(cmyk5 - cmyk3))
            print 'representation error:', reperr
        return reperr

# assumes same mode for both images
def concatenate_horizontally(fd1, fd2):
    '''
    Input: Two file descriptors (from im.open()) for images; must be
    of the same mode.

    Output: File descriptor for a new image with the two images placed
    side-by-side.
    '''
    w = fd1.size[0] + fd2.size[0]
    h = np.max([fd1.size[1], fd2.size[1]])
    assert(fd1.mode == fd2.mode)
    result = im.new(fd1.mode, (w, h))
    result.paste(fd1, (0, 0))
    result.paste(fd2, (fd1.size[0], 0))
    return result

# assumes same mode for both images
def concatenate_vertically(fd1, fd2):
    '''
    Same as concatenate_horizontally() but output with the two images
    stacked one on top of the other.
    '''
    w = np.max([fd1.size[0], fd2.size[0]])
    h = fd1.size[1] + fd2.size[1]
    result = im.new(fd1.mode, (w, h))
    result.paste(fd1, (0, 0))
    result.paste(fd2, (0, fd1.size[1]))
    return result

def test_logistic():
    '''
    Test code for logistic() and inverse_logistic().
    '''
    x0 = np.random.uniform(size=100000)
    q1 = logistic(x0)
    x1 = inverse_logistic(q1)
    q2 = logistic(x1)
    x2 = inverse_logistic(q2)
    worst1 = np.argmax(np.abs(x0-x1))
    print 'test_logistic worst x1', x1[worst1], (x1-x0)[worst1]
    worst2 = np.argmax(np.abs(x0-x2))
    print 'test_logistic worst x2', x2[worst2], (x2-x0)[worst2]
    worst3 = np.argmax(np.abs(q1-q2))
    print 'test_logistic worst q2', q2[worst3], (q1-q2)[worst3]
    return None

def test_float2byte():
    '''
    Test code for hoggprinter.float2byte() and hoggprinter.byte2float().
    '''
    hp = hoggprinter(0.02, 0.15, 0.1)
    b0 = range(256)
    f1 = hp.byte2float(b0)
    b1 = np.array(hp.float2byte(f1))
    f2 = hp.byte2float(b1)
    worst1 = np.argmax(np.abs(b0-b1))
    print 'test_float2byte worst b1', b1[worst1], (b1-b0)[worst1]
    worst2 = np.argmax(np.abs(f1-f2))
    print 'test_float2byte worst f2', f2[worst2], (f2-f1)[worst2]
    return None

def test_rgb2cmyk():
    '''
    Test code for hoggprinter.rgb2cmyk() and hoggprinter.cmyk2rgb().
    '''
    hp = hoggprinter(0.02, 0.15, 0.1) # realistic
    # hp = hoggprinter(0.01, 0.01, 0.01) # optimistic
    # hp = hoggprinter(0., 0., 0.) # unreal
    for rgb in (np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([0., 0., 1.]),
                np.array([0., 0., 0.]),
                np.array([0.499, 0.499, 0.499]), # 0.5 is a bad number; why?
                np.array([1., 1., 1.]),
                np.random.uniform(size=(3,))):
        rgbc = np.clip(rgb, 0.002, 0.998)
        reperr = hp.test(rgbc)
    maxreperr = 0.
    ntrials = 100
    print 'worst RGB triple found among %d random trials:' % ntrials
    for i in range(ntrials):
        rgb = np.random.uniform(size=(3,))
        reperr = hp.test(rgb, verbose=False)
        if reperr > maxreperr:
            maxreperr = reperr
            worstrgb = rgb
    reperr = hp.test(worstrgb, verbose=True)
    return None

def test_image():
    '''
    Convert one image from RGB to CMYK.  This function can serve as
    example code and produces example output.
    '''
    ifn = 'test.jpg'
    ofn = 'bar.tiff'
    hp = hoggprinter(0.02, 0.15, 0.1)
    hp.rgb2cmyk_image_file(ifn, ofn)
    print hp
    return None

def test_strip():
    '''
    Take an input test RGB image and convert it to CMYK on a grid of
    different printer settings to build a "test strip" that can be
    printed and measured or inspected.

    There is some straightforward concatenate logic in this code, and
    it produces tons of intermediate files.  That is because I am
    impatient when the grass is growing.
    '''
    import pylab as plt
    ifn = 'test.jpg'
    tmpfn = 'foo.png'
    ifd = im.open(ifn, mode='r').rotate(90)
    nx, ny = ifd.size
    k = 0
    dKfd = None
    dKrange = [0.2, 0.1, 0.05]
    ddrange = [0.02, 0.01, 0.005]
    dorange = [0.02, 0.01, 0.005]
    for dK in dKrange:
        ddfd = None
        for dd in ddrange:
            dofd = None
            for do in dorange:
                k += 1
                plt.clf()
                plt.figure(figsize=(1., 0.2))
                plt.axes([0., 0., 1., 1.])
                plt.xlim(0,1)
                plt.ylim(0,1)
                plt.axis('off')
                plt.text(0,0,'%.3f %.3f %.3f' % (dK, dd, do))
                plt.savefig(tmpfn, dpi = nx)
                tmpfd = im.open(tmpfn, mode='r').convert('RGB')
                cfd = concatenate_vertically(tmpfd, ifd)
                cfd.save('foo-%03d.png' % k)
                hp = hoggprinter(dK, dd, do)
                ofd = hp.rgb2cmyk_image(cfd)
                if dofd is None:
                    dofd = ofd
                else:
                    dofd = concatenate_horizontally(dofd, ofd)
            if ddfd is None:
                ddfd = dofd
            else:
                ddfd = concatenate_vertically(ddfd, dofd)
            ddfd.save('bar-bar-%03d.tiff' % k)
        if dKfd is None:
            dKfd = ddfd
        else:
            dKfd = concatenate_horizontally(dKfd, ddfd)
        dKfd.save('bar-bar-bar-%03d.tiff' % k)
    return None

def main():
    if True:
        test_strip()
    if False:
        test_logistic()
        test_float2byte()
        test_rgb2cmyk()
        test_image()
    return None

if __name__ == '__main__':
    import cProfile as cp
    cp.run('main()')
