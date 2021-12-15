
################################################################################
# -- Simulating Exc-Inh spiking networks in response to inhibitory perturbation
################################################################################

import numpy as np; import pylab as pl; import time, os, pickle
from scipy.stats import norm
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *;
import networkTools; reload(networkTools); import networkTools as net_tools
import nest

cwd = os.getcwd()

t_init = time.time()

################################################################################
#### functions

# -- rectification
def _rect_(xx): return xx*(xx>0)

# -- generates the weight matrix
def _mycon_(N1, N2, B12, pr=1.):
    zb = np.random.binomial(1, pr, (N1,N2))
    zw = np.sign(B12) * _rect_(np.random.normal(abs(B12),abs(B12)/5,(N1,N2)))
    zz = zb* zw
    return zz

# -- runs a network simulation with a defined inh perturbation
bw = 50.
def myRun(rr1, rr2, Tstim=Tstim, Tblank=Tblank, Ntrials=Ntrials, bw = bw, \
            rec_conn={'EtoE':1, 'EtoI':1, 'ItoE':1, 'ItoI':1}, nn_stim=0):

        # -- restart the simulator
        net_tools._nest_start_()

        init_seed = np.random.randint(1, 1234, n_cores)
        nest.SetStatus([0],[{'rng_seeds':init_seed.tolist()}])

        # -- exc & inh neurons
        exc_neurons = net_tools._make_neurons_(NE, neuron_model=cell_type, \
        myparams={'b':NE*[0.], 'a':NE*[0.]})
        inh_neurons = net_tools._make_neurons_(NI, neuron_model=cell_type, \
        myparams={'b':NE*[0.],'a':NE*[0.]})

        all_neurons = exc_neurons + inh_neurons

        # -- recurrent connectivity
        if rec_conn['EtoE']:
            net_tools._connect_pops_(exc_neurons, exc_neurons, W_EtoE)
        if rec_conn['EtoI']:
            net_tools._connect_pops_(exc_neurons, inh_neurons, W_EtoI)
        if rec_conn['ItoE']:
            net_tools._connect_pops_(inh_neurons, exc_neurons, W_ItoE)
        if rec_conn['ItoI']:
            net_tools._connect_pops_(inh_neurons, inh_neurons, W_ItoI)

        # -- recording spike data
        spikes_all = net_tools._recording_spikes_(neurons=all_neurons)

        # -- background input
        pos_inp = nest.Create("poisson_generator", N)

        for ii in range(N):
            nest.Connect([pos_inp[ii]], [all_neurons[ii]], \
            syn_spec = {'weight':Be_bkg, 'delay':delay_default})

        # -- simulating network for N-trials
        for tri in range(Ntrials):
            print('')
            print('# -> trial # ', tri+1)

            ## transient
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr1[ii]})
            net_tools._run_simulation_(Ttrans)

            ## baseline
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr1[ii]})
            net_tools._run_simulation_(Tblank)

            ## perturbing a subset of inh
            for ii in range(N):
                nest.SetStatus([pos_inp[ii]], {'rate':rr2[ii]})
            net_tools._run_simulation_(Tstim)

        # -- reading out spiking activity
        spd = net_tools._reading_spikes_(spikes_all)

        # -- computes the rates out of spike data in a given time interval
        def _rate_interval_(spikedata, T1, T2, bw=bw):
            tids = (spikedata['times']>T1) * (spikedata['times']<T2)
            rr = np.histogram2d(spikedata['times'][tids], spikedata['senders'][tids], \
                 range=((T1,T2),(1,N)), bins=(int((T2-T1)/bw),N))[0] / (bw/1e3)
            return rr

        rout_blank = np.zeros((Ntrials, int(Tblank / bw), N))
        rout_stim = np.zeros((Ntrials, int(Tstim / bw), N))
        for tri in range(Ntrials):
            Tblock = Tstim+Tblank+Ttrans
            rblk = _rate_interval_(spd, Tblock*tri+Ttrans, Tblock*tri+Ttrans+Tblank)
            rstm = _rate_interval_(spd, Tblock*tri+Ttrans+Tblank, Tblock*(tri+1))
            rout_blank[tri,:,:] = rblk
            rout_stim[tri,:,:] = rstm

        print('##########')
        print('## Mean firing rates {Exc | Inh (pert.) | Inh (non-pert.)}')
        print('## Before pert.: ', \
        np.round(rout_blank[:,:,0:NE].mean(),1), \
        np.round(rout_blank[:,:,NE:NE+nn_stim].mean(),1), \
        np.round(rout_blank[:,:,NE+nn_stim:].mean(),1) )
        print('## After pert.: ', \
        np.round(rout_stim[:,:,0:NE].mean(),1), \
        np.round(rout_stim[:,:,NE:NE+nn_stim].mean(),1), \
        np.round(rout_stim[:,:,NE+nn_stim:].mean(),1) )
        print('##########')

        return rout_blank, rout_stim, spd

################################################################################

for ij1, Be in enumerate(Be_rng):
    for ij2, Bi in enumerate(Bi_rng):

        Bee, Bei = Be, Be
        Bie, Bii = Bi, Bi

        print('####################')
        print('### (Be, Bi): ', Be, Bi)
        print('####################')

        # -- result path
        res_path = cwd+'/SimulationResults/'
        if not os.path.exists(res_path): os.mkdir(res_path)

        os.chdir(res_path)

        # -- L23 recurrent connectivity
        W_EtoE = _mycon_(NE, NE, Bee, .15)
        W_EtoI = _mycon_(NE, NI, Bei, .15)
        W_ItoE = _mycon_(NI, NE, Bie, 1.)
        W_ItoI = _mycon_(NI, NI, Bii, 1.)

        # -- running simulations
        sim_res = {}

        for nn_stim in nn_stim_rng:

            print('\n # -----> size of pert. inh: ', nn_stim)

            r_extra = np.zeros(N)
            r_extra[NE:NE+nn_stim] = r_stim

            rr1 = r_bkg*np.ones(N)
            rr2 = rr1 + r_extra

            sim_res[nn_stim] = myRun(rr1, rr2, nn_stim=nn_stim)

        sim_res['nn_stim_rng'], sim_res['Ntrials'] = nn_stim_rng, Ntrials
        sim_res['N'], sim_res['NE'], sim_res['NI'] = N, NE, NI
        sim_res['Tblank'], sim_res['Tstim'], sim_res['Ttrans'] = Tblank, Tstim, Ttrans

        os.chdir(res_path);
        sim_name = 'sim_res_Be'+str(Be)+'_Bi'+str(Bi)
        fl = open(sim_name, 'wb'); pickle.dump(sim_res, fl); fl.close()

os.chdir(cwd)

t_end = time.time()
print('took: ', np.round((t_end-t_init)/60), ' mins')

################################################################################
################################################################################
################################################################################
