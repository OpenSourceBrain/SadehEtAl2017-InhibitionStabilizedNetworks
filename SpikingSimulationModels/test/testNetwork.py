################################################################################
# -- Testing the network to facilitate translation to PyNN
################################################################################

import numpy as np
import sys
from imp import reload

sys.path.append("..")

import defaultParams
reload(defaultParams)
from defaultParams import *
import networkTools; reload(networkTools); import networkTools as net_tools
import nest






# -- rectification
def _rect_(xx): return xx*(xx>0)

# -- generates the weight matrix
def _mycon_(N1, N2, B12, pr=1.):
    zb = np.random.binomial(1, pr, (N1,N2))
    zw = np.sign(B12) * _rect_(np.random.normal(abs(B12),abs(B12)/5,(N1,N2)))
    zz = zb* zw
    return zz





def testNetwork(Be, Bi , nn_stim, show_gui=True):

        Ntrials = 1
        bw = 50.
        
        rec_conn={'EtoE':1, 'EtoI':1, 'ItoE':1, 'ItoI':1}

        print('####################')
        print('### (Be, Bi, nn_stim): ', Be, Bi, nn_stim)
        print('####################')
        
        Bee, Bei = Be, Be
        Bie, Bii = Bi, Bi

        print('\n # -----> size of pert. inh: ', nn_stim)

        r_extra = np.zeros(N)
        r_extra[NE:NE+nn_stim] = r_stim

        rr1 = r_bkg*np.ones(N)
        rr2 = rr1 + r_extra

        
        
        # -- restart the simulator
        net_tools._nest_start_()

        init_seed = np.random.randint(1, 1234, n_cores)
        nest.SetStatus([0],[{'rng_seeds':init_seed.tolist()}])

        # -- exc & inh neurons
        exc_neurons = net_tools._make_neurons_(NE, neuron_model=cell_type, \
        myparams={'b':NE*[0.], 'a':NE*[0.]})
        inh_neurons = net_tools._make_neurons_(NI, neuron_model=cell_type, \
        myparams={'b':NE*[0.],'a':NE*[0.]})
        
        # -- L23 recurrent connectivity
        W_EtoE = _mycon_(NE, NE, Bee, .15)
        W_EtoI = _mycon_(NE, NI, Bei, .15)
        W_ItoE = _mycon_(NI, NE, Bie, 1.)
        W_ItoI = _mycon_(NI, NI, Bii, 1.)
        
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
                 range=((T1,T2),(1,N)), bins=((T2-T1)/bw,N))[0] / (bw/1e3)
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
        
        
        from pyneuroml import pynml
        xs = [[],[],[]]
        ys = [[],[],[]]
        all_e = []
        all_ip = []
        all_inp = []
        for i in range(len(spd['senders'])):
            cellnum = spd['senders'][i]
            time = spd['times'][i]
            if cellnum<NE:
                xs[0].append(time)
                all_e.append(time)
                ys[0].append(cellnum)
            elif cellnum<NE+nn_stim:
                xs[1].append(time)
                all_ip.append(time)
                ys[1].append(cellnum)
            else:
                xs[2].append(time)
                all_inp.append(time)
                ys[2].append(cellnum)
        
        if show_gui:
            
            print("Plotting %s spikes for %s E cells, %s spikes for %s Ip cells, %s spikes for %s Inp cells"%(len(xs[0]), NE ,len(xs[1]), nn_stim, len(xs[2]), N-NE-nn_stim))

            pynml.generate_plot(xs,
                                ys,
                                "Spike times: Be=%s; Bi=%s; N=%s; p=%s"%(Be,Bi,N,nn_stim), 
                                xaxis = "Time (s)", 
                                yaxis = "Cell number", 
                                colors = ['red','black','blue'],
                                linestyles = ['','',''],
                                markers = ['.','.','.'],
                                markersizes = [1,1,1],
                                grid = False,
                                show_plot_already=False)

            plt.figure()
            bins = 15
            plt.hist(all_e, bins=bins,histtype='step',weights=[1/float(NE)]*len(all_e),color='red')
            plt.hist(all_ip, bins=bins,histtype='step',weights=[1/float(nn_stim)]*len(all_ip),color='black')
            plt.hist(all_inp, bins=bins,histtype='step',weights=[1/float(N-NE-nn_stim)]*len(all_inp),color='blue',ls='--')
            plt.title("Histogram of spikes")
        
        
    
if __name__ == '__main__':
    
    Be=0.1
    Bi=-0.2
    
    nn_stim_rng = (np.array([0.1, .25, .5, .75, 1])*NI).astype('int')
    nn_stim_rng = (np.array([0.1,.75])*NI).astype('int')

    if '-nogui' in sys.argv:
        show_gui = False
    else:
        import matplotlib.pyplot as plt 
        show_gui = True 


    for nn_stim in nn_stim_rng:
        testNetwork(Be, Bi, nn_stim, show_gui=show_gui)
        
    if show_gui:
        plt.show()