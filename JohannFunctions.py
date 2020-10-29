import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# For final fits
from iminuit import Minuit
from ExternalFunctions import Chi2Regression

# For primary guess fit
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

# Roc-curve related
from sklearn.metrics import auc

def double_gauss_fit(mass, bins = 100, range = (400, 600), ax = None, verbose = True, guesses = None, max_size = None, plimit = 0, color = "red",lambd = False):
    # Fit third degree polynomium to background
    if not lambd:
        def background_fit(x, a, b, c, d):
            res = a * (x- 498) ** 3 + b * (x-498) ** 2 + c * (x-498) + d
            return res
    else:
        def background_fit(x, a, b, c, d,e,f):
            xx = x- 1116.5
            if(x<0):
                res = a * (xx) ** 3 + b * (xx) ** 2 + c * (xx) + d
            else:
                res = e*xx+f
            return res
    # The double gauss signal
    def add_signal(x, mean, sig, size, ratio, sig_ratio):
        return size * binwidth *  (ratio * norm.pdf(x, mean, sig) + \
                                   (1 - ratio) * norm.pdf(x, mean, sig_ratio * sig))
    
    # The full fit
    if not lambd:
        def full_fit(x, mean, sig, size, ratio, sig_ratio, a, b, c, d):
            return background_fit(x, a, b, c, d) + add_signal(x, mean, sig, size, ratio, sig_ratio)
    else:
        def full_fit(x, mean, sig, size, ratio, sig_ratio, a, b, c, d,e,f):
            return background_fit(x, a, b, c, d,e,f) + add_signal(x, mean, sig, size, ratio, sig_ratio)
        
    # Make histogram
    vals, edges = np.histogram(mass, bins = bins, range = range)
    xs = (edges[1:] + edges[:-1])/2

    binwidth = xs[1] - xs[0]

    mask = vals > 0
    vals = vals[mask]  
    xs = xs[mask]
    errs = np.sqrt(vals)

    # Get guesses for a background fit
    if not guesses:
        back_data_mask = abs(xs - xs[np.argmax(vals)]) > 10
        
        if not lambd:
            background_guess =  [0, 0, (vals[-1]- vals[0]) / 100, np.mean(vals)]
        else:
            background_guess =  [0, 0, (vals[-1]- vals[0]) / 100, np.mean(vals),5,100]
            
        if len(vals[back_data_mask]) == 0:
            return None, None, None, None

        try: 
            vals_b, cov_b = curve_fit(background_fit, xs[back_data_mask], vals[back_data_mask], p0 = background_guess)
        except:
            vals_b = background_guess
        if not lambd:
            b1, b2, b3, b4 = vals_b
        
            bkgr_chi2 = Chi2Regression(background_fit, xs[back_data_mask], vals[back_data_mask], errs[back_data_mask])
            bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4)
        else:
            b1, b2, b3, b4,b5,b6 = vals_b
        
            bkgr_chi2 = Chi2Regression(background_fit, xs[back_data_mask], vals[back_data_mask], errs[back_data_mask])
            bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4,e=b5,f=b6)
        bkgr_min.migrad()
        counter = 0
        while not bkgr_min.valid and counter<50:
            bkgr_min.migrad()
            counter += 1
        if not bkgr_min.valid: print("No background valid minimum found!")
        
        #Save guesses 
        if not lambd:
            b1, b2, b3, b4 = bkgr_min.args
        else:
            b1, b2, b3, b4,b5,b6 = bkgr_min.args
            

        guesses_sig = [498, 7, 2000, 0.5, 2]
        if lambd:
            guesses_sig = [1116.5, 2, 2000, 0.5, 2]
        try:
            vals_f, cov_f = curve_fit(full_fit, xs, vals, p0 = guesses_sig + [b1, b2, b3, b4])
        except:
            vals_f = np.hstack([guesses_sig, vals_b])
        
        if not lambd:
            s1, s2, s3, s4, s5, b1, b2, b3, b4 = vals_f
        else:
            s1, s2, s3, s4, s5, b1, b2, b3, b4,b5,b6 = vals_f
            
    else:
        if not lambd:
            s1, s2, s3, s4, s5, b1, b2, b3, b4 = guesses
        else:
            s1, s2, s3, s4, s5, b1, b2, b3, b4,b5,b6 = guesses
            

    full_chi2 = Chi2Regression(full_fit, xs, vals, errs)
    if not lambd:
        full_min  = Minuit(full_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4, \
                           mean = s1, sig = s2, size = s3, ratio = s4, sig_ratio = s5, limit_sig_ratio = (1, 4), \
                           limit_ratio = (0, 1.0), limit_mean = (490, 510), limit_size = (0, max_size), limit_sig = (3, 10))
    else:
        full_min  = Minuit(full_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4,e=b5,f=b6, \
                           mean = s1, sig = s2, size = s3, ratio = s4, sig_ratio = s5, limit_sig_ratio = (1, 4), \
                           limit_ratio = (0, 1.0), limit_mean = (1110, 11120), limit_size = (0, max_size), limit_sig = (3, 10))
    
    full_min.migrad()
    
    full_min.migrad()
    counter = 0
    while not full_min.valid and counter<2:
        full_min.migrad()
        counter += 1
    if not full_min.valid: print("No valid minimum found!")

    # Check fit
    chi = full_min.fval 
    pval = chi2.sf(chi, np.sum(mask) - len(full_min.args))

    if verbose:
        print(f"Completed fit with Chi2: {chi:.1f}, p-val: {pval:.3f} and the total amount of signal " + \
            f"{full_min.values['size']:.0f} +/- {full_min.errors['size']:.0f}, background: {len(mass) - int(full_min.values['size'])}")


    if ax:
        ax.plot(xs, vals, alpha = 1, color = color)
#         ax.errorbar(xs, vals, errs, elinewidth = 1, color = 'k', capsize = 2, linestyle = 'none', alpha = 0.25)
#         ax.plot(xs, full_fit(xs, *full_min.args), '--', alpha = 0.5)

    if True:#full_min.errors['size'] < full_min.values['size'] and full_min.valid and pval > plimit:
        return full_min.values['size'], len(mass) - full_min.values['size'], full_min.errors['size'], full_min.args
    else:
        return None, None, None, None


def roc_curve_data(mass, probs, Npoints = 10, bins = 100, range = (400, 600), ax_roc = None , ax_fits = None, verbose = True, plimit = 0.01, ax_hist = None):
    sigs, bkgrs, errs = [], [], []
    mass = np.array(mass)
    mass = mass[np.argsort(probs)]
    cuts = (len(mass) / Npoints * np.arange(0, Npoints)).astype(int)
    args = None
    max_size = None
    from matplotlib.cm import winter
    colors = winter(np.linspace(0, 1, Npoints)[::-1])
    from scipy.special import logit
    lprobs = np.sort(logit(probs))
    if ax_hist:
        n, edges, patches = ax_hist.hist(lprobs, bins = bins, histtype = 'stepfilled', color = 'gray')
#         print(n)
        for cut, color in zip(cuts, colors):
            ax_hist.vlines(lprobs[cut], 0, max(n), color = color, linestyle = "dashed", alpha = 0.5)
            
    for i, c in zip(cuts, colors):
        bkgr, sig, err, args = double_gauss_fit(mass[i:], bins = bins, range = range, ax = ax_fits, verbose = verbose, guesses = args, max_size = max_size, plimit = plimit, color = c)
        if bkgr:
            bkgrs.append(bkgr)
            sigs.append(sig)
            errs.append(err)
            if len(sigs) == 1:
                max_size = 1 * args[2]

    sigs, bkgrs, errs = np.array(sigs), np.array(bkgrs), np.array(errs)
    y = sigs/sigs.max()
    x = bkgrs/bkgrs.max()

#     x = np.append(x, 0)[::-1]
#     y = np.append(y, 0)[::-1]

    AUC_estimate = np.trapz(np.append(x, 0), np.append(y, 0))

#     x_errs = np.sqrt((errs/sigs.max()) ** 2 + (errs[sigs.argmax()] * sigs / sigs.max() ** 2) ** 2)
#     y_errs = np.sqrt((errs/bkgrs.max()) ** 2 + (errs[bkgrs.argmax()] * bkgrs / bkgrs.max() ** 2) ** 2)
    
    if ax_roc:
        ax_roc.scatter(x, y, c = colors)
        ax_roc.vlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
        ax_roc.hlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
#         ax_roc.errorbar(x[:-1], y[:-1], x_errs, y_errs, elinewidth = 1, capsize = 2, color = 'k', ls = 'none')
        ax_roc.set(xlim = (-0.2, 1.2), ylim = (-0.2, 1.2))
    
    return AUC_estimate


def roc_curve_data_lambda(mass, probs, Npoints = 10, bins = 100, range = (1100, 1140), ax_roc = None , ax_fits = None, verbose = True, plimit = 0.01, ax_hist = None):
    sigs, bkgrs, errs = [], [], []
    mass = np.array(mass)
    mass = mass[np.argsort(probs)]
    cuts = (len(mass) / Npoints * np.arange(0, Npoints)).astype(int)
    args = None
    max_size = None
    from matplotlib.cm import winter
    colors = winter(np.linspace(0, 1, Npoints)[::-1])
    from scipy.special import logit
    lprobs = np.sort(logit(probs))
    if ax_hist:
        n, edges, patches = ax_hist.hist(lprobs, bins = bins, histtype = 'stepfilled', color = 'gray')
#         print(n)
        for cut, color in zip(cuts, colors):
            ax_hist.vlines(lprobs[cut], 0, max(n), color = color, linestyle = "dashed", alpha = 0.5)
            
    for i, c in zip(cuts, colors):
        bkgr, sig, err, args = double_gauss_fit(mass[i:], bins = bins, range = range, ax = ax_fits, verbose = verbose, guesses = args, max_size = max_size, plimit = plimit, color = c,lambd = True)
        if bkgr:
            bkgrs.append(bkgr)
            sigs.append(sig)
            errs.append(err)
            if len(sigs) == 1:
                max_size = 1 * args[2]

    sigs, bkgrs, errs = np.array(sigs), np.array(bkgrs), np.array(errs)
    y = sigs/sigs.max()
    x = bkgrs/bkgrs.max()

#     x = np.append(x, 0)[::-1]
#     y = np.append(y, 0)[::-1]

    AUC_estimate = np.trapz(np.append(x, 0), np.append(y, 0))

#     x_errs = np.sqrt((errs/sigs.max()) ** 2 + (errs[sigs.argmax()] * sigs / sigs.max() ** 2) ** 2)
#     y_errs = np.sqrt((errs/bkgrs.max()) ** 2 + (errs[bkgrs.argmax()] * bkgrs / bkgrs.max() ** 2) ** 2)
    
    if ax_roc:
        try:
            ax_roc.scatter(x, y, c = colors)
        except:
            ax_roc.scatter(x, y, c = colors[0])
        ax_roc.vlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
        ax_roc.hlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
#         ax_roc.errorbar(x[:-1], y[:-1], x_errs, y_errs, elinewidth = 1, capsize = 2, color = 'k', ls = 'none')
        ax_roc.set(xlim = (-0.2, 1.2), ylim = (-0.2, 1.2))
    
    return AUC_estimate
    


