'''
This file is part of PrinterModel.
Copyright 2012 David W. Hogg (NYU) <http://cosmo.nyu.edu/hogg/>.

PrinterModel is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License, version 2, as
published by the Free Software Foundation.

PrinterModel is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
version 2 along with this program.  If not, see
<http://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
'''

# to-do
# -----
# - better initialization for rgb2cmyk
# - proper doc strings for many functions
# - proper unit tests for logistic and inverse
# - proper unit tests for float2byte and inverse
# - proper unit tests for rgb2cmyk and inverse

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

    Initialize with three parameters:  YADA, YADA, YADA

    Note:  Internal hard-set magic number epsilon.
    '''
    epsilon = 0.001 # magic number

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
        Input: 3-element ndarray of RGB values on [0, 1] for a single
        pixel.

        Output: 4-element ndarray of CMYK values on [0, 1] for a single
        image pixel.

        Internally, this code works by optimizing a chi-squared-like
        objective function based on the forward function cmyk2rgb().
        The objective function weakly prefers using K ink with the
        self.epsilon parameter.
        '''
        def chi(pars):
            cmyk = logistic(pars)
            return np.append(rgb - self.cmyk2rgb(cmyk), self.epsilon * (1. - cmyk[3]))
        pars = inverse_logistic(self.rgb2cmyk_stoopid(rgb))
        bestpars, foo = op.leastsq(chi, pars, maxfev=3000)
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

        Note that floats on [0, 1] and bytes on [0, 255] are related by
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
        ofd2 = im.new('CMYK', ifd.size)
        odata2 = [self.rgb2cmyk_bytes(d) for d in ifd.getdata()]
        ofd2.putdata(odata2)
        return ofd2

    def rgb2cmyk_image_file(self, ifn, ofn):
        ifd = im.open(ifn, mode='r').convert('RGB') # this gets rid of alpha channel!
        ofd2 = self.rgb2cmyk_image(ifd)
        ofd2.save(ofn)
        return None

    def test(self, rgb, verbose=True):
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
    w = fd1.size[0] + fd2.size[0]
    h = np.max([fd1.size[1], fd2.size[1]])
    result = im.new(fd1.mode, (w, h))
    result.paste(fd1, (0, 0))
    result.paste(fd2, (fd1.size[0], 0))
    return result

# assumes same mode for both images
def concatenate_vertically(fd1, fd2):
    w = np.max([fd1.size[0], fd2.size[0]])
    h = fd1.size[1] + fd2.size[1]
    result = im.new(fd1.mode, (w, h))
    result.paste(fd1, (0, 0))
    result.paste(fd2, (0, fd1.size[1]))
    return result

def test_logistic():
    x0 = np.random.uniform(size=1000000)
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

def main_one_pixel():
    # realistic
    hp = hoggprinter(0.02, 0.15, 0.1)
    # optimistic
    # hp = hoggprinter(0.01, 0.01, 0.01)
    # unreal
    # hp = hoggprinter(0., 0., 0.)
    for rgb in (np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([0., 0., 1.]),
                np.array([0., 0., 0.]),
                np.array([0.5, 0.5, 0.5]),
                np.array([1., 1., 1.]),
                np.random.uniform(size=(3,))):
        reperr = hp.test(rgb)
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

def main_image():
    ifn = 'test.jpg'
    ofn = 'bar.tiff'
    hp = hoggprinter(0.02, 0.15, 0.1)
    hp.rgb2cmyk_image_file(ifn, ofn)
    print hp
    return None

def main_test_strip():
    import pylab as plt
    ifn = 'test.jpg'
    tmpfn = 'foo.png'
    ifd = im.open(ifn, mode='r').rotate(90)
    nx, ny = ifd.size
    k = 0
    dKfd = None
    drange = [0.3, 0.03, 0.003]
    for dK in drange:
        ddfd = None
        for dd in drange:
            dofd = None
            for do in drange:
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
                dofd.save('bar-%03d.tiff' % k)
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
    if False:
        main_one_pixel()
        main_image()
        main_test_strip()
    if True:
        test_logistic()
        test_float2byte()
    return None

if __name__ == '__main__':
    import cProfile as cp
    cp.run('main()')
