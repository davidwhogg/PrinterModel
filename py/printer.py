# This file is part of PrinterModel.
# Copyright 2012 David W. Hogg (NYU).

# PrinterModel is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License, version 2, as
# published by the Free Software Foundation.

# PrinterModel is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# version 2 along with this program.  If not, see
# <http://www.gnu.org/licenses/old-licenses/gpl-2.0.html>

# to-do
# -----
# - proper doc strings
# - convert np arrays to sequences much more smartly?
# - structure / context for image conversion?
# - better ways to do cacheing of converstions? kd-tree or db or full 16M array?

import numpy as np
import scipy.optimize as op
import Image as im

def logistic(x):
    return 1. / (1. + np.exp(-x))

class hoggprinter():

    # a small number; DWH intuits that it should be < 1/256
    epsilon = 0.001

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

    # input 4-element np.array on [0, 1] (ie, one pixel)
    # output 3-element np.array on [0, 1]
    def cmyk2rgb(self, cmyk):
        return np.prod(1. - self.eta * cmyk, axis=1)

    # input 3-element np.array on [0, 1] (ie, one pixel)
    # output 4-element np.array on [0, 1]
    def rgb2cmyk(self, rgb):
        def chi(pars):
            chi = np.zeros(4)
            cmyk = logistic(pars)
            chi[0:3] = rgb - self.cmyk2rgb(cmyk)
            chi[3] = self.epsilon * (1. - cmyk[3])
            return chi
        pars = np.zeros(4)
        bestpars, foo = op.leastsq(chi, pars, maxfev=3000)
        return logistic(bestpars)

    # input 3-element sequence of bytes (ie, one real image pixel)
    # output 4-element sequence of bytes
    def rgb2cmyk_bytes(self, rgb_bytes):
        self.conversions += 1
        if (self.conversions % 1024) == 0:
            print self.conversions, self
        try:
            cmyk_bytes = self.cache[rgb_bytes]
        except KeyError:
            cmyk = self.rgb2cmyk((np.array(rgb_bytes).astype(float) + 0.5) / 256.)
            tmp = np.clip((np.array(cmyk) * 256.).astype(int), 0, 255)
            cmyk_bytes = (tmp[0], tmp[1], tmp[2], tmp[3])
            self.cache[rgb_bytes] = cmyk_bytes
        return cmyk_bytes

    # input: input and output filenames
    def rgb2cmyk_image(self, ifd):
        ofd2 = im.new('CMYK', ifd.size)
        odata2 = [self.rgb2cmyk_bytes(d) for d in ifd.getdata()]
        ofd2.putdata(odata2)
        return ofd2

    # input: input and output filenames
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
    if True:
        main_image()
    if False:
        main_test_strip()
    return None

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':8})
    rc('text', usetex=True)
    import pylab as plt
    import cProfile as cp
    cp.run('main()')
