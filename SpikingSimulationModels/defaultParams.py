
################################################################################
# -- Default set of parameters
################################################################################

import numpy as np; import pylab as pl; import time, sys, os
import matplotlib

## the number of cores to be used for simulations
n_cores = 4

# define the NEST path if it's needed
#nest_path = '/Users/sadra/NEST/nest/ins/lib/python3.4/site-packages/'
#nest_path = '/Users/sadra/NEST/lib/python3.5/site-packages/'
#if os.path.exists(nest_path):
#    sys.path.append(nest_path)

#------------- neuron params

# resting potential (mV)
Ur = -70.e-3
# reversal potential of exc. (mV)
Ue = 0.e-3
# reversal potential of inh. (mV)
Ui = -75.e-3
# threshold voltage (mV)
Uth = -50.e-3
# reset potential (mV)
Ureset = -60.e-3

# membrane capacitance (F)
C = 120e-12
# leak conductance (S)
Gl = 1./140e6

# sample Exc and Inh conductances (nS)
Be, Bi = .1, -.2

# range of Exc and Inh conductances (nS)
Be_rng = np.array([0.01, .05, .1, .15, .2, .25])
Bi_rng = np.array([-.1, -.2, -.3, -.4, -.5])

# background and stimulus conductances (nS)
Be_bkg = .1
Be_stim = .1

# exc. synaptic time constant (s)
tau_e = 1.e-3
# inh. synaptic time constant (s)
tau_i = 1.e-3

# refractory time (s)
t_ref = 2e-3

# conductance-based alpha-synapses neuron model
neuron_params_default = \
{'C_m': C*1e12,
  'E_L': Ur*1000.,
  'E_ex': Ue*1000.,
  'E_in': Ui*1000.,
  'I_e': 0.0, # 130.0 for slow spiking...
  'V_m': Ur*1000.,
  'V_reset': Ureset*1000.,
  'V_th': Uth*1000.,
  'g_L': Gl*1e9,
  't_ref': t_ref*1000.,
  'tau_syn_ex': tau_e*1000.,
  'tau_syn_in': tau_i*1000.}

# -- simulation params

#default synaptic delay (ms)
delay_default = .1

# time resolution of simulations (ms)
dt = .1

# background rate (sp/s)
r_bkg = 10000.-400.
# rate of perturbation (sp/s)
r_stim = -400.

# transitent time to discard the data (ms)
Ttrans = 500.
# simulation time before perturbation (ms)
Tblank= 500.
# simulation time of perturbation (ms)
Tstim = 500.
# time after perturbation
Tpost = 500

# number of trials
Ntrials = 5

# -- network params

# N: total population size (Exc + Inh)
# frac: fraction of Inh neurons
def set_total_population_size(N_total, frac = .2):
    global N
    global NE
    global NI
    N = N_total
    # size of Inh population
    NI = int(frac*N)
    # size of Exc population
    NE = N - NI
    print("Setting %s cells total, %s E cells, %s I cells"%(N,NE,NI))
    
if not 'N' in locals():
    set_total_population_size(2000)
    

# range of the size of Inh perturbations
nn_stim_rng = (np.array([0.1, .25, .5, .75, 1])*NI).astype('int')

# single cell type
cell_type = 'aeif_cond_alpha'

# -- default settings for plotting figures
# (comment out for conventional Python format)
matplotlib.rc('font', serif='sans-serif')

SIZE = 15
try:
    pl.rc('font', size=SIZE)  # controls default text sizes
    pl.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    pl.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    pl.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    pl.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    pl.rc('legend', fontsize=SIZE)  # legend fontsize
    pl.rc('figure', titlesize=SIZE)  # fontsize of the figure title
except:
    pass # Just cosmetic...

# half-frame axes
def HalfFrame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

################################################################################
################################################################################
################################################################################
