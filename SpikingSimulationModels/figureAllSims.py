
################################################################################
# -- Plots the results of simulations for the entire range of parameter space
################################################################################

import numpy as np; import pylab as pl; import os, pickle
import matplotlib as mplt
from scipy.interpolate import interp1d
from parula import parula_map
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *

cwd = os.getcwd()

# -- reading the results
os.chdir(cwd+'/SimulationResults')

fl = open('analysis_results', 'rb'); analysis_results=pickle.load(fl); fl.close()
[rdIp_all, nns_finer, zip_finer_all,  nn_crit_all] = analysis_results

# -- plotting
fig_path = cwd+'/Figures/'
if not os.path.exists(fig_path): os.mkdir(fig_path)
os.chdir(fig_path)

ner, nir = len(Be_rng), len(Bi_rng)
dr_mean = np.zeros((ner,nir))
dr_min, dr_max = np.zeros(dr_mean.shape), np.zeros(dr_mean.shape)

# flags for plotting figures
plot_diff_rates = 1
plot_crit_fraction = 1

# --
if plot_diff_rates:
    fig = pl.figure(figsize=(14,7))

    newax = pl.subplot(111)

    for ij1, Be in enumerate(Be_rng):
        for ij2, Bi in enumerate(Bi_rng):
            nn_crit = nn_crit_all[ij1,ij2]

            ymn = np.nanmin(rdIp_all[Be,Bi])
            ymx = np.nanmax(rdIp_all[Be,Bi])

            ax = pl.subplot(nir, ner, ner*(nir-ij2-1)+ij1+1)
            HalfFrame(ax)

            ax.plot(nn_stim_rng, rdIp_all[Be,Bi], 'o', ms=10, mew=2.5, mec='b', mfc='none')

            drt = zip_finer_all[Be, Bi]
            pl.plot(nns_finer, drt, 'b-', lw=3)

            dr_mean[ij1,ij2] = np.nanmean(drt)
            dr_min[ij1,ij2] = np.nanmin(drt)
            dr_max[ij1,ij2] = np.nanmax(drt)

            pl.plot(nns_finer, np.zeros(len(nns_finer)), 'k-', lw=1)
            pl.plot([0, NE/2, NE], np.zeros(3), 'k|', ms=5)

            pl.plot([nn_crit, nn_crit], [np.round(ymn-1),np.round(ymx+1)], 'r--', lw=2)

            if ij1 == 0:
               if ij2 == 0:
                   ax.set_xlabel(r'$p/N_I$ (%)', size=12.5)

                   ax.set_ylabel('Diff. rate (sp/s)', size=12.5)

            ax.set_xticks(np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])*NI)
            ax.set_xlim([-10,NI+20])

            if ij1 == 0 and ij2 == 0:
                ax.set_xticklabels([0, '', '', '', '', 50, '', '', '', '', 100])
            else:
                ax.set_xticklabels([])
            ax.set_yticks([np.round(ymn-1),0,np.round(ymx+1)])
            ax.set_yticklabels([int(np.round(ymn-1)),0,int(np.round(ymx+1))])

    ## axes ticks
    newax = fig.add_axes([.125,.2, .85,.75])
    newax.patch.set_visible(False)

    for spinename, spine in newax.spines.items():
        if spinename == 'top' or spinename == 'right':
            spine.set_visible(False)

    newax.spines['bottom'].set_position(('outward', 50))
    newax.spines['left'].set_position(('outward', 50))
    newax.spines['right'].set_position(('outward', 50))
    newax.spines['top'].set_position(('outward', 50))

    Be_rng_new = np.copy(Be_rng)
    newax.set_xticks(Be_rng_new)
    newax.set_xticklabels(Be_rng)
    dd = np.diff(Be_rng)[0]
    newax.set_xlim(Be_rng_new[0]-dd/3, Be_rng_new[-1]+dd/3)
    newax.set_xlabel('Be (nS)', size=20)

    newax.set_yticks(-Bi_rng)
    newax.set_yticklabels(-Bi_rng)
    dd = np.diff(-Bi_rng)[0]
    newax.set_ylim(-Bi_rng[0]-dd/3, -Bi_rng[-1]+dd/3)
    newax.set_ylabel('Bi (nS)', size=20)

    newax.tick_params(axis='both', which='major', labelsize=20)

    pl.subplots_adjust(wspace=.25, hspace=.25, bottom=.2, top=.95, left=.12, right=.975)

    pl.savefig('figure_AllSims_ratesDiff.pdf')

# --
if plot_crit_fraction:

    dff = np.diff(Be_rng)/2
    Be_rng2 = Be_rng- np.array(dff.tolist()+[dff[-1]])
    Be_rng2 = np.array(Be_rng2.tolist()+[Be_rng2[-1]+np.diff(Be_rng2)[-1]])

    dff = np.diff(Bi_rng)/2
    Bi_rng2 = Bi_rng- np.array(dff.tolist()+[dff[-1]])
    Bi_rng2 = np.array(Bi_rng2.tolist()+[Bi_rng2[-1]+np.diff(Bi_rng2)[-1]])

    xmesh, ymesh = np.meshgrid(Be_rng2, abs(Bi_rng2))

    fig = pl.figure(figsize=(12,6))

    zzz = nn_crit_all/NI*100
    label = r'$p/N_I$ (%)'

    ax = pl.subplot(111)

    Zm = np.ma.masked_where(np.isnan(zzz),zzz)

    pl.pcolormesh(xmesh, ymesh, Zm.T, cmap=parula_map, vmin=0, vmax=100)

    cbaxes = fig.add_axes([0.125, 0.7, 0.02, 0.25])
    cb = pl.colorbar(cax = cbaxes)

    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.set_label(label, size=15)

    ax.set_xticks([])
    ax.set_yticks([])

    ## axes ticks
    newax = ax
    newax.patch.set_visible(False)

    for spinename, spine in newax.spines.items():
        if spinename == 'top' or spinename == 'right':
            spine.set_visible(False)

    newax.spines['bottom'].set_position(('outward', 20))
    newax.spines['left'].set_position(('outward', 20))
    newax.spines['right'].set_position(('outward', 50))
    newax.spines['top'].set_position(('outward', 50))

    newax.set_xticks(Be_rng)
    newax.set_xticklabels(Be_rng)
    dd = np.diff(Be_rng)[-1]
    newax.set_xlabel('Be (nS)', size=20)

    newax.set_yticks(-Bi_rng)
    newax.set_yticklabels(-Bi_rng)
    dd = np.diff(-Bi_rng)[0]
    newax.set_ylabel('Bi (nS)', size=20)

    newax.tick_params(axis='both', which='major', labelsize=20)

    pl.subplots_adjust(wspace=.5, hspace=.35, bottom=.175, top=.95, left=.1, right=.975)

    pl.savefig('figure_AllSims_critFraction.pdf')

os.chdir(cwd)

pl.show()

################################################################################
################################################################################
################################################################################
