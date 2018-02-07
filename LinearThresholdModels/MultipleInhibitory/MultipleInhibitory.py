
import pylab as pl; import numpy as np; 
import os; import pickle; 
from scipy.interpolate import interp1d

# ----------------------------------------------

# -- generate random connectivity (normal) 
# -- Input:
# mw: mean weight; sf: fraction of mean-weight as the std of the normal distribution
# n1, n2: number of pre- and post-synaptic neurons 
# -- Output:
# weight matrix of connectivity between pre- and post-synaptic neurons
def _rancon_(mw, sf, n1, n2):
    return _rect_(np.random.normal(mw, mw*sf, (n1, n2)))

# -- half-wave rectification
def _rect_(z): return z * (z>0)

# -- rate-based solver of the network dynamics
# -- Input:
# N: number of units; W: weight matrix; Trng: total time-steps of simulation
# stim_rate: stimulus rate to each unit; threshold: threshold of linear-threshold units 
# -- Ouput:
# output rates of all units during the simulation time 
def _rateSolver_(N, W, Trng, stim_rate, threshold):
    r = np.zeros(N)
    rall = []
    for it, t in enumerate(Trng):
        #if it % 1000 == 0: print(it)

        I = _rect_( np.array(np.matrix(r) * np.matrix(W))[0] + stim_rate - threshold)
        dr = dt/tau * (-r + I)

        r = r + dr
        rall.append(r)
        
    return np.array(rall)

# ----------------------------------------------

# -- network params

NE = 400 # no. of Exc / PC neurons
NP = 50 # no. of PV neurons
NS = 25 # no. of SOM neurons
NV = 25 # no. of VIP neurons

# total no. of neurons
N = NE + NP + NS + NV

# recurrent coupling to parameterize the weight matrix
J = 1.5
# inhibition dominance in the PC-PV network
g = 1

# -- default params of the rate-based solver
T = 200 # total time of simulation
dt = .1 # time resolution of simulation
tau = 20. # time constant of neuronal integration

Trng = np.arange(0, T, dt)

threshold = 0*np.ones(N) # threshold of individual neurons

# -- params of simulated perturbation 

# the size of perturbation 
pert_size = -.2

# fraction of inhibitory subpopulation to perturb
pert_frac_range = np.array([.1, .3, .5, .7, .9, 1])

# coupling of PC<->SOM network
som_couplings = [0, .1, .2, .3, .4, .5, .6, .7, .8]
som_couplings = som_couplings[::2]

# which subclass of interneurons to perturb
pert_PV = 1
pert_SOM = 0#1
pert_VIP = 1

sim_id = 'pert_PV'+str(pert_PV)+'SOM'+str(pert_SOM)+'VIP'+str(pert_VIP)

# -- organizing the results

# class of rates simulated before (R0) & after perturbation (R)
class c_R0:
     def __init__(self):
        self.pc, self.pv_pert, self.pv_nonpert = [], [], []
        self.som_pert, self.som_nonpert, self.vip = [], [], []
class c_R:
     def __init__(self):
        self.pc, self.pv_pert, self.pv_nonpert = [], [], []
        self.som_pert, self.som_nonpert, self.vip = [], [], []

R0 = c_R0()
R = c_R()

# -- control flags
run_sims = 1
plot_figs = 1

# --------------------------------------------------------
# -- running simulations

if run_sims:
    for som_coupling in som_couplings:

        print('######## SOM coupling: ', som_coupling)

        # -- building the weight matrix
        stdf = .2
        Wee = _rancon_(J/NE, stdf, NE, NE)
        Wep = _rancon_(J/NE, stdf, NE, NP)
        Wes = _rancon_(1/NE, stdf, NE, NS) *som_coupling
        Wev = _rancon_(1/NE, stdf, NE, NV)

        Wpe = _rancon_(g*J/NP, stdf, NP, NE) * -1
        Wpp = _rancon_(g*J/NP, stdf, NP, NP) * -1
        Wps = _rancon_(1/NP, stdf, NP, NS) * 0.
        Wpv = _rancon_(1/NP, stdf, NP, NV) * 0.

        Wse = _rancon_(1/NS, stdf, NS, NE) * -1 *som_coupling
        Wsp = _rancon_(1/NS, stdf, NS, NP) * -.5 
        Wss = _rancon_(1/NS, stdf, NS, NS) * 0.
        Wsv = _rancon_(1/NS, stdf, NS, NS) * -.6

        Wve = _rancon_(1/NV, stdf, NV, NE) * 0
        Wvp = _rancon_(1/NV, stdf, NV, NP) * 0
        Wvs = _rancon_(1/NV, stdf, NV, NS) * -.25
        Wvv = _rancon_(1/NV, stdf, NV, NS) * 0

        W = np.concatenate((np.concatenate((Wee, Wep, Wes, Wev), 1), \
                            np.concatenate((Wpe, Wpp, Wps, Wpv), 1), \
                            np.concatenate((Wse, Wsp, Wss, Wsv), 1), \
                            np.concatenate((Wve, Wvp, Wvs, Wvv), 1)
                            ), 0)

        
        r0_pc, r0_pv_pert = [], []
        r_pc, r_pv_pert = [], []
        Rates, Rates0 = [], []

        for pert_frac in pert_frac_range:

            print('# -- pert frac: ', pert_frac)

            ###
            #print('(before perturbation)')
            stim_rate = np.ones(N)
            
            rall0 = _rateSolver_(N, W, Trng, stim_rate, threshold)
            Rates0.append(rall0)
            
            #converg = np.mean(np.diff(np.mean(rall0,1))[-100:])
            #print(converg)
            
            r0_pc.append(np.nanmean(rall0[-100:,0:NE]))
            r0_pv_pert.append(np.nanmean(rall0[-100:,NE:int(NE+pert_frac*NP)]))            

            ###
            #print('(after perturbation)')

            if pert_PV:
               stim_rate[NE:int(NE+pert_frac*NP)] += pert_size
            if pert_SOM:
               stim_rate[NE+NP:int(NE+NP+pert_frac*NS)] += pert_size
            if pert_VIP:
               stim_rate[NE+NP+NS:int(NE+NP+NS+pert_frac*NV)] += pert_size
            
            rall = _rateSolver_(N, W, Trng, stim_rate, threshold)
            Rates.append(rall)
            
            #converg = np.mean(np.diff(np.mean(rall,1))[-100:])
            #print(converg)
            
            r_pc.append(np.nanmean(rall[-100:,0:NE]))
            r_pv_pert.append(np.nanmean(rall[-100:,NE:int(NE+pert_frac*NP)]))
            
        Rates0 = np.array(Rates0); Rates = np.array(Rates)
        
        # --
        
        R0.pc.append(np.array(r0_pc))
        R0.pv_pert.append(np.array(r0_pv_pert))

        R.pc.append(np.array(r_pc))
        R.pv_pert.append(np.array(r_pv_pert))

    results = [R, R0, Rates, Rates0]
    fl = open('Results_'+sim_id, 'wb'); pickle.dump(results, fl); fl.close()
else:
    fl = open('Results_'+sim_id, 'rb'); results = pickle.load(fl); fl.close()
    [R, R0, Rates, Rates0] = results

# --------------------------------------------------------
### plotting figure

# -- formatting figures
pl.style.use('seaborn-white')

SIZE = 14
pl.rc('font', size=SIZE)  # controls default text sizes
pl.rc('axes', titlesize=SIZE)  # fontsize of the axes title
pl.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
pl.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
pl.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
pl.rc('legend', fontsize=SIZE)  # legend fontsize
pl.rc('figure', titlesize=SIZE)  # fontsize of the figure title
pl.rc('xtick.major', size=5)
pl.rc('ytick.major', size=5)
pl.rc('xtick.major', width=1)
pl.rc('ytick.major', width=1)

if plot_figs:
    # -- interpolating the critical fraction of inh needed for the paradoxical effect
    zz = np.array(R.pv_pert) - np.array(R0.pv_pert)

    crit_fracs = np.zeros(len(som_couplings))
    xx_interp = np.arange(pert_frac_range.min(), pert_frac_range.max(), .01)
    for i, som_coupling in enumerate(som_couplings):
        fint = interp1d(pert_frac_range, zz[i])
        zz_interp = fint(xx_interp)
        crit_id = np.mean(np.where( (zz_interp <= .01) * (zz_interp >= -.01))[0])
        if np.isnan(crit_id):
            crit_frac = np.nan
        else:
            crit_frac = xx_interp[crit_id]
        crit_fracs[i] = np.round(100*crit_frac)

    #
    pl.figure(figsize=(6,5))

    ax = pl.subplot(111)
    axt = ax.twinx()

    pl.title(sim_id, y=1.02)

    ax.plot(som_couplings, crit_fracs, 'b-o')
    ax.set_ylabel('Thresh. prop. of pert. (%)', color='b')
    ax.set_ylim([50, 100])

    axt.plot(som_couplings, np.nanmean(R0.pc,1), 'r-o')
    axt.set_ylabel('Mean Exc. activity (a.u.)', color='r')
    axt.set_ylim([0-.1, 1.5])
    axt.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])
    axt.set_yticklabels([0, '', .2, '', .4, '', .6, '', .8, '', 1, '', 1.2, '', 1.4, ''])

    ax.set_xlabel('PC-SOM coupling')
    ax.set_xticks([0, .2, .4, .6, .8])
    ax.set_xticklabels([0, .2, .4, .6, .8])

    ax.set_xlim([-.02, .85])

    pl.subplots_adjust(left=.15, right=.85, bottom=.15)

    pl.savefig('Fig_'+sim_id+'.pdf')

pl.show()
