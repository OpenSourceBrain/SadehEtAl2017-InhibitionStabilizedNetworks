
################################################################################
# -- Plots the results of simulations for a sample network
################################################################################

import numpy as np; import pylab as pl; import os, pickle
import matplotlib.font_manager as font_manager
from scipy.interpolate import interp1d
from parula import parula_map
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *

cwd = os.getcwd()

# -- reading the results
os.chdir(cwd+'/SimulationResults')
sim_name = 'sim_res_Be'+str(Be)+'_Bi'+str(Bi)
fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()

fl = open('analysis_results', 'rb'); analysis_results=pickle.load(fl); fl.close()
[rdIp_all, nns_finer, zip_finer_all,  nn_crit_all] = analysis_results

# -- plotting

# flags for plotting figures
plot_raster_plots = 1
plot_population_rates = 1
plot_rates_avg = 1
plot_differential = 1

fig_path = cwd+'/Figures/'
if not os.path.exists(fig_path): os.mkdir(fig_path)
os.chdir(fig_path)

# -- sample values of pert. inh. to plot
my_rng = (np.array([.1, .75])*NI).astype('int')
ll = len(my_rng)
Tm = Ttrans+Tblank; dT = Tblank

# --
if plot_raster_plots:
    pl.figure(figsize=(12,6))

    tri = 0
    for ii, nn_stim in enumerate(my_rng):
        print(nn_stim)
        spd = sim_res[nn_stim][2]

        ax = pl.subplot(2,1,ii+1);

        exc_ids = (spd['senders'] <= NE)
        inh_ids_pert = (spd['senders'] > NE) * (spd['senders'] <= NE+nn_stim)
        inh_ids_Npert = (spd['senders'] > NE+nn_stim)

        pl.plot(spd['times'][exc_ids], spd['senders'][exc_ids], 'r|', label='Exc.')
        pl.plot(spd['times'][inh_ids_pert], spd['senders'][inh_ids_pert], 'b|', label='pert. Inh.')
        pl.plot(spd['times'][inh_ids_Npert], spd['senders'][inh_ids_Npert], 'c|', label='non-pert. Inh.')

        ax.tick_params(axis='both', which='major', labelsize=15)

        if ii == 0:
            ax.text(-0.075, 0.35, 'Exc.',
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes, color='red', fontsize=12.5)
            ax.text(-0.075, 0.7, 'Inh. \n(pert.)',
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes, color='b', fontsize=12.5)
            ax.text(-0.075, 0.9, 'Inh.\n(non-pert.)',
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes, color='c', fontsize=12.5)

        ax.set_ylim(0-200,N+100)
        ax.set_xlim([Tm-dT, Tm+dT])

        pl.plot([Tm,Tm], [0, N], 'k--', lw=2)

        if ii == 0:
            ax.text(Tm+dT-50, -400, '100 ms', fontsize=12.5, \
            verticalalignment='bottom', horizontalalignment='center')
            ax.plot([Tm+dT-100, Tm+dT], [-200, -200], 'k-', lw=15)

        pl.fill_between([Tm-dT,Tm], [-200, -200], [-10, -10], color='gray', alpha=.25)
        pl.fill_between([Tm,Tm+dT], [-200, -200], [-10, -10], color='orange', alpha=1)

        ax.text(Tm-dT/2, -125, 'Normal', fontsize=12.5, \
        verticalalignment='center', horizontalalignment='center')
        ax.text(Tm+dT/2, -125, 'Perturbation ('+str(int(100*nn_stim/NI))+'%)', \
        fontsize=12.5, verticalalignment='center', horizontalalignment='center')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_frame_on(0)

        if ii == 2:
            pl.xlabel('Time (ms)', size=20);

    pl.subplots_adjust(bottom=.05, right=.95, top=.95)

    pl.savefig('figure_SampleNet_rasterPlot.pdf')

# --
if plot_population_rates:

    bw = 50
    tt_blank = np.arange(0, Tblank, bw)
    tt_stim = np.arange(Tblank, Tblank+Tstim, bw)

    pl.figure(figsize=(8,6))

    for ii, nn_stim in enumerate(my_rng):

        rblank_all = sim_res[nn_stim][0]
        rstim_all = sim_res[nn_stim][1]

        ax = pl.subplot(2,1,ii+1);
        pl.title('Perturbation ('+str(int(100*nn_stim/NI))+'%)', size=20, loc='right')

        zz = rblank_all
        tt = tt_blank + bw/2

        pl.plot(tt, np.mean(np.mean(zz[:,:,0:NE],2),0), 'r-', lw=2)
        pl.plot(tt, np.mean(np.mean(zz[:,:,NE:NE+nn_stim],2),0), 'b-', lw=2)
        pl.plot(tt, np.mean(np.mean(zz[:,:,NE+nn_stim:],2),0), 'c--', lw=2)

        ##

        zz = rstim_all
        tt = tt_stim + bw/2

        zz1m = np.mean(np.mean(zz[:,:,0:NE],2),0)
        zz2m = np.mean(np.mean(zz[:,:,NE:NE+nn_stim],2),0)
        zz3m = np.mean(np.mean(zz[:,:,NE+nn_stim:],2),0)

        pl.plot(tt, zz1m, 'r-', lw=2, label='Exc.')
        pl.plot(tt, zz2m, 'b-', lw=2, label='Inh. (pert.)')
        pl.plot(tt, zz3m, 'c--', lw=2, label='Inh. (non-pert.)')

        if ii == 0:
            pl.legend(loc=2, frameon=0, fontsize=12.5)
        else:
            pl.xlabel('Time (s)', size=20)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_yticklabels([0, '', 4, '', 8, ''])
        pl.plot([Tstim,Tstim], [0, 10.1], 'k--', lw=2)

        ax.set_xticks([0, dT, 2*dT])
        ax.set_xticklabels([0, np.round(dT/1e3,1), np.round(2*dT/1e3,1)])

        ax.spines['left'].set_position(('outward', 10))

        if ii == 1:
            pl.ylabel('                    Population rate (/s)', size=20)

        ax.tick_params(axis='both', which='major', labelsize=20)
        HalfFrame(ax)

    pl.subplots_adjust(hspace=.5, bottom=.15)

    pl.savefig('figure_SampleNet_popRates.pdf')

# --
if plot_rates_avg:
    rblank_avg = []
    rstim_avg = []
    for ii, nn_stim in enumerate(nn_stim_rng):
        rblank_avg.append(np.mean(np.mean(sim_res[nn_stim][0],0),0))
        rstim_avg.append(np.mean(np.mean(sim_res[nn_stim][1],0),0))
    rblank_avg = np.array(rblank_avg)
    rstim_avg = np.array(rstim_avg)

    pl.figure()
    ax = pl.subplot(111);

    for ii, nn_stim in enumerate(nn_stim_rng):
        xx = nn_stim/NI*100*np.ones(Ntrials)

        zz1 = np.mean(np.mean(sim_res[nn_stim][1],1)[:,0:NE],1) \
            - np.mean(np.mean(sim_res[nn_stim][0],1)[:,0:NE],1)
        zz2 = np.mean(np.mean(sim_res[nn_stim][1],1)[:,NE:NE+nn_stim],1) \
            - np.mean(np.mean(sim_res[nn_stim][0],1)[:,NE:NE+nn_stim],1)
        zz3 = np.mean(np.mean(sim_res[nn_stim][1],1)[:,NE+nn_stim:],1) \
            - np.mean(np.mean(sim_res[nn_stim][0],1)[:,NE+nn_stim:],1)

        pl.plot(xx, zz1, \
        'o', ms=7.5, mew=1, mec='r', mfc='r', alpha=.25)
        pl.plot(xx, zz2, \
        'o', ms=7.5, mew=1, mec='b', mfc='b', alpha=.25)
        pl.plot(xx, zz3, \
        'o', ms=7.5, mew=1, mec='c', mfc='c', alpha=.25)


        pl.plot(nn_stim/NI*100, np.mean(zz1), \
        'r_', ms=20, mew=3, label='Exc.')
        pl.plot(nn_stim/NI*100, np.mean(zz2), \
        'b_', ms=20, mew=3, label='Inh. (pert.)')
        pl.plot(nn_stim/NI*100, np.mean(zz3), \
        'c_', ms=20, mew=3, label='Inh. (non-pert.)')

        pl.plot([0, 100], [0, 0], 'k--', lw=2)

        if ii == 0:
            pl.legend(frameon=0, loc=2, fontsize=20, numpoints=1)

    pl.xlabel(r'$p/N_I$ (%)', size=20)
    pl.ylabel('Diff. firing rate (/s)', size=20)

    xtks = np.arange(0,110,10)
    xtklbls = []
    for ti, xt in enumerate(xtks):
        if xt not in nn_stim_rng/NI*100:
            xtklbls.append('')
        else:
            xtklbls.append(int(xt))
    ax.set_xticks(xtks)
    ax.set_xticklabels(xtklbls)

    ax.tick_params(axis='both', which='major', labelsize=20)
    HalfFrame(ax)

    pl.xlim(0-2,100+2)
    pl.ylim(-1.5)

    pl.subplots_adjust(bottom=.15)

    pl.savefig('figure_SampleNet_meanRates.pdf')

# --
if plot_differential:
    ### difference function
    zE0 = np.zeros((Ntrials, len(nn_stim_rng)))
    zI0_pert, zI0_Npert = np.zeros(zE0.shape), np.zeros(zE0.shape)
    for ii, nn_stim in enumerate(nn_stim_rng):
        zE0[:,ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,0:NE],1),1) \
                - np.mean(np.mean(sim_res[nn_stim][0][:,:,0:NE],1),1)
        zI0_pert[:,ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,NE:NE+nn_stim],1),1) \
                    - np.mean(np.mean(sim_res[nn_stim][0][:,:,NE:],1),1)
        zI0_Npert[:,ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,NE+nn_stim:],1),1) \
                        - np.mean(np.mean(sim_res[nn_stim][0][:,:,NE:],1),1)

    zE = np.mean(zE0,0)
    zI_pert = np.mean(zI0_pert,0)
    zI_Npert = np.mean(zI0_Npert,0)

    pl.figure()
    ax = pl.subplot(111)

    pl.plot(nn_stim_rng/NI*100, zE, 'r-o', label='Exc.')
    pl.plot(nn_stim_rng/NI*100, zI_pert, 'b-o', label='Inh. (pert.)')
    pl.plot(nn_stim_rng/NI*100, zI_Npert, 'c-o', label='Inh. (non-pert.)')

    pl.fill_between(nn_stim_rng/NI*100, \
                    zE-np.std(zE0,0), zE+np.std(zE0,0), \
                    color='r', alpha=.15)

    pl.fill_between(nn_stim_rng/NI*100, \
                    zI_pert-np.std(zI0_pert,0), zI_pert+np.std(zI0_pert,0), \
                    color='b', alpha=.15)

    pl.fill_between(nn_stim_rng/NI*100, \
                    zI_Npert-np.std(zI0_Npert,0), zI_Npert+np.std(zI0_Npert,0), \
                    color='c', alpha=.15)

    pl.legend(frameon=0, loc=2, numpoints=1, fontsize=20)

    nns_finer = np.arange(nn_stim_rng[0]/NI*100, nn_stim_rng[-1]/NI*100, 1)
    zip_func = interp1d(nn_stim_rng/NI*100, zI_pert, 'linear')
    zip_finer = zip_func(nns_finer)

    pl.plot(nns_finer, zip_finer, 'b--')

    zzz = (zip_finer<.1)[0:-1] * (zip_finer>=0)[0:-1] * (np.diff(zip_finer)>0)
    if sum(zzz) > 0:
        nn_crit = nns_finer[int(np.mean(np.where(zzz == 1)[0]))]
    else: nn_crit = np.nan

    pl.plot(nn_crit, zip_func(nn_crit), 'x', ms=20, mew=2.5)
    ymn,ymx = -1, 2
    pl.plot([nn_crit, nn_crit], [ymn,ymx], 'r--', lw=2)
    pl.plot(nns_finer, np.zeros(len(nns_finer)), 'k--', lw=2)

    xtks = np.arange(0,110,10)
    xtklbls = []
    for ti, xt in enumerate(xtks):
        if xt not in nn_stim_rng/NI*100:
            xtklbls.append('')
        else:
            xtklbls.append(int(xt))
    ax.set_xticks(xtks)
    ax.set_xticklabels(xtklbls)

    pl.xlim(0-2, 100+2)

    pl.xlabel(r'$p/N_I$ (%)', size=20)
    pl.ylabel('Diff. firing rate (/s)', size=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    HalfFrame(ax)

    pl.subplots_adjust(bottom=.15)

    pl.savefig('figure_SampleNet_ratesDiff.pdf')

os.chdir(cwd)

pl.show()

################################################################################
################################################################################
################################################################################
