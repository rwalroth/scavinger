# -*- coding: utf-8 -*-

# Python imports

# Other imports
import numpy as np

# This package imports

def gaus(x, mu, sig):
    """
    normalized gaussian distribution, sig = full width half max,
    mu = energy at max
    """
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp((-0.5)*((x-mu)/sig)**2)


def lore(x, mu, sig):
    """
    normalized lorentzian distribution, sig = full width half max,
    mu = energy at max
    """
    return ((1/np.pi)*(sig/((x - mu)**2 + sig**2)))


def psvo(x, mu, sig, amp, alpha):
    """
    pseudo voigt approximated as convolution of gaussian and lorentzian line
    shapes. sig and mu same as above, amp is the area, alpha the degree of
    lorentzian character
    """
    sigg = sig/(np.sqrt(2*np.log(2)))
    return ((1-alpha)*amp*gaus(x, sigg, mu)) + (alpha*amp*lore(x, sig, mu))


def psvo2(x, mu, sig, amp, alpha):
    """
    pseudo voigt with constant inflection points as opposed to constant FWHM
    """
    sigg = np.sqrt(3)*sig
    return ((1-alpha)*amp*gaus(x, sig, mu)) + (alpha*amp*lore(x, sigg, mu))


def psvo3(x, mu, sig, hght, alpha):
    """
    Pseudo-Voigt that takes peak height instead of area as a variable.
    """
    n = np.sqrt(2*np.log(2))
    amp = hght/((1 - alpha)/(sig*np.sqrt(2*np.pi)/n)+alpha/(sig*np.pi))
    return psvo(x, mu, amp, sig, alpha)


def edge(x, e0, w, amp):
    """
    arctangential edge function. e0 defines inflection point, w the broadening,
    amp the peak height
    """
    return amp*(np.arctan((x - e0)/w) + np.pi/2)/np.pi


def edge2(x, e0, w, amp, alpha):
    """
    stepwise edge broadened with pseudo-voigt lineshapes. e0 is the inflection,
    w the broadening, alpha the same as for psvo.
    """
    mu = e0 + w
    n = np.sqrt(2*np.log(2))
    amp2 = amp/((1 - alpha)/(w*np.sqrt(2*np.pi)/n)+alpha/(w*np.pi))
    if x <= mu - w:
        a = psvo(x, mu, amp2, w, alpha)
    elif x > mu - w:
        a = psvo(mu, mu, amp2, w, alpha) - psvo(x, (mu - 2*w), amp2, w, alpha)
    return a


def edge3(x, e0, w, amp, alpha):
    """
    allows edge2 to handle arrays and lists in addition to ints and floats
    """
    mu = e0 + w
    n = np.sqrt(2*np.log(2))
    amp2 = amp/((1 - alpha)/(w*np.sqrt(2*np.pi)/n)+alpha/(w*np.pi))
    if type(x) == np.ndarray:
        a = np.array([edge2(i, e0, amp, w, alpha) for i in x])
    else:
        if x <= mu - w:
            a = psvo(x, mu, amp2, w, alpha)
        elif x > mu - w:
            a = (psvo(mu, mu, amp2, w, alpha)
                 - psvo(x, (mu - 2*w), amp2, w, alpha))
    return a


def multigaus(x, *p):
    """
    sum of many gaussian peaks, *p is a list divisible by 3 with each set
    of 3 numbers corresponding to mu, sig, amp
    """
    val = 0
    n = 0
    while (n + 2) < len(p):
        val += p[n+2]*gaus(x, p[n], p[n+1])
        n += 4
    return val


def multilore(x, *p):
    """
    sum of many lorentzian peaks, *p is a list divisible by 3 with each set
    of 3 numbers corresponding to mu, sig, amp
    """
    val = 0
    n = 0
    while (n + 2) < len(p):
        val += p[n+2]*lore(x, p[n], p[n+1])
        n += 4
    return val


def multipsvo(x, *p):
    """
    sum of many pseudo voigt peaks, *p is a list divisible by 4 with each set
    of 4 numbers corresponding to psvo variables
    """
    val = 0
    n = 0
    while (n + 3) < len(p):
        val += psvo(x, p[n], p[n+1], p[n+2], p[n+3])
        n += 4
    return val


def xanes(x, *p):
    """
    sum of edge3 and multipeak. First four variables used by edge3, rest by
    multipeak.
    """
    val = 0
    val += edge3(x, p[0], p[1], p[2], p[3])
    n = 4
    while (n + 3) < len(p):
        val += psvo(x, p[n], p[n+1], p[n+2], p[n+3])
        n += 4
    return val