
import sys
import numpy as np

from pyNN.utility import get_script_args, Timer, ProgressBar
from pyNN.random import NumpyRNG, RandomDistribution
from neo.io import PyNNTextIO

sys.path.append("../SpikingSimulationModels")

import defaultParams


def runNetwork(Be, 
               Bi, 
               nn_stim, 
               show_gui=True,
               dt = defaultParams.dt, 
               N_rec_v = 5, 
               save=False, 
               simtime = defaultParams.Tstim+defaultParams.Tblank+defaultParams.Ttrans, 
               extra = {}):
    
    exec("from pyNN.%s import *" % simulator_name) in globals()
    
    timer = Timer()

    rec_conn={'EtoE':1, 'EtoI':1, 'ItoE':1, 'ItoI':1}

    print('####################')
    print('### (Be, Bi, nn_stim): ', Be, Bi, nn_stim)
    print('####################')

    Bee, Bei = Be, Be
    Bie, Bii = Bi, Bi

    N = defaultParams.N
    NE = defaultParams.NE
    NI = defaultParams.NI

    print('\n # -----> size of pert. inh: ', nn_stim)

    r_extra = np.zeros(N)
    r_extra[NE:NE+nn_stim] = defaultParams.r_stim

    rr1 = defaultParams.r_bkg*np.ones(N)
    rr2 = rr1 + r_extra
    
    
    rank = setup(timestep=dt, max_delay=defaultParams.delay_default, **extra)
    
    print("rank =", rank)
    nump = num_processes()
    print("num_processes =", nump)
    import socket
    host_name = socket.gethostname()
    print("Host #%d is on %s" % (rank+1, host_name))

    if 'threads' in extra:
        print("%d Initialising the simulator with %d threads..." % (rank, extra['threads']))
    else:
        print("%d Initialising the simulator with single thread..." % rank)
        
        
    timer.start()  # start timer on construction
    
    print("%d Setting up random number generator" % rank)
    kernelseed = 123
    rng = NumpyRNG(kernelseed, parallel_safe=True)
    
    '''
    # conductance-based alpha-synapses neuron model
    neuron_params_default = \
    {'C_m': C*1e12,
      'E_L': Ur*1000.,
      'E_ex': Ue*1000.,
      'E_in': Ui*1000.,
      'I_e': 0.0,
      'V_m': Ur*1000.,
      'V_reset': Ureset*1000.,
      'V_th': Uth*1000.,
      'g_L': Gl*1e9,
      't_ref': t_ref*1000.,
      'tau_syn_ex': tau_e*1000.,
      'tau_syn_in': tau_i*1000.}'''
    
    nesp = defaultParams.neuron_params_default
    cell_parameters = {
        'cm':         nesp['C_m']/1000,   # Capacitance of the membrane in nF
        'tau_refrac': nesp['t_ref'],     # Duration of refractory period in ms.
        'v_spike':    0.0 ,     # Spike detection threshold in mV.   https://github.com/nest/nest-simulator/blob/master/models/aeif_cond_alpha.cpp
        'v_reset':    nesp['V_reset'],     # Reset value for V_m after a spike. In mV.
        'v_rest':     nesp['E_L'],     # Resting membrane potential (Leak reversal potential) in mV.
        
        
        
        
        
        'tau_m':      16.8,  # Membrane time constant in ms
        
        
        
        
        
        
        
        'i_offset':   nesp['I_e']/1000,     # Offset current in nA
        'a':          0,     # Subthreshold adaptation conductance in nS.
        'b':          0,  # Spike-triggered adaptation in nA
        'delta_T':    2 ,     # Slope factor in mV. See https://github.com/nest/nest-simulator/blob/master/models/aeif_cond_alpha.cpp
        'tau_w':      144.0,     # Adaptation time constant in ms. See https://github.com/nest/nest-simulator/blob/master/models/aeif_cond_alpha.cpp
        'v_thresh':   nesp['V_th'],     # Spike initiation threshold in mV
        'e_rev_E':    nesp['E_ex'],     # Excitatory reversal potential in mV.
        'tau_syn_E':  nesp['tau_syn_ex'],     # Rise time of excitatory synaptic conductance in ms (alpha function).
        'e_rev_I':    nesp['E_in'],     # Inhibitory reversal potential in mV.
        'tau_syn_I':  nesp['tau_syn_in'],     # Rise time of the inhibitory synaptic conductance in ms (alpha function).
    }

    print("%d Creating excitatory population with %d neurons." % (rank, NE))
    celltype = EIF_cond_alpha_isfa_ista(**cell_parameters)
    celltype.default_initial_values['v'] = cell_parameters['v_rest'] # Setting default init v, useful for NML2 export
    E_net = Population(NE, celltype, label="E")
    
    p_rate = defaultParams.r_bkg
    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_type = SpikeSourcePoisson(rate=p_rate)
    expoisson = Population(NE, source_type, label="expoisson")
    
    
    
    progress_bar = ProgressBar(width=20)
    #connector = FixedProbabilityConnector(epsilon, rng=rng, callback=progress_bar)
    #E_syn = StaticSynapse(weight=JE, delay=delay)
    #I_syn = StaticSynapse(weight=JI, delay=delay)
    ext_Connector = OneToOneConnector(callback=progress_bar)
    ext_syn = StaticSynapse(weight=0.0001, delay=dt)
    
    
    input_to_E = Projection(expoisson, E_net, ext_Connector, ext_syn, receptor_type="excitatory")
    print("input --> E\t", len(input_to_E), "connections")
    
    
    # Record spikes
    print("%d Setting up recording in excitatory population." % rank)
    E_net.record('spikes')
    if N_rec_v>0:
        E_net[0:min(NE,N_rec_v)].record('v')
    
    
    # read out time used for building
    buildCPUTime = timer.elapsedTime()
    # === Run simulation ===========================================================

    # run, measure computer time
    timer.start()  # start timer on construction
    print("%d Running simulation in %s for %g ms (dt=%sms)." % (rank, simulator_name, simtime, dt))
    run(simtime)
    print("Done")
    simCPUTime = timer.elapsedTime()
    
    # write data to file
    if save and not simulator_name=='neuroml':
        for pop in [E_net]:
            io = PyNNTextIO(filename="ISN-%s-%s-%i.gdf"%(simulator_name, pop.label, rank))
            spikes =  pop.get_data('spikes', gather=False)
            for segment in spikes.segments:
                io.write_segment(segment)
                
            io = PyNNTextIO(filename="ISN-%s-%s-%i.dat"%(simulator_name, pop.label, rank))
            vs =  pop.get_data('v', gather=False)
            for segment in vs.segments:
                io.write_segment(segment)
            
    spike_data = {}
    spike_data['senders'] = []
    spike_data['times'] = []
    index_offset = 1
    for pop in [E_net]:
        if rank == 0:
            spikes =  pop.get_data('spikes', gather=False)
            #print(spikes.segments[0].all_data)
            num_rec = len(spikes.segments[0].spiketrains)
            print("Extracting spike info (%i) for %i cells in %s"%(num_rec,pop.size,pop.label))
            #assert(num_rec==len(spikes.segments[0].spiketrains))
            for i in range(num_rec):
                ss = spikes.segments[0].spiketrains[i]
                for s in ss:
                    index = i+index_offset
                    #print("Adding spike at %s in %s[%i] (cell %i)"%(s,pop.label,i,index))
                    spike_data['senders'].append(index)
                    spike_data['times'].append(s)
            index_offset+=pop.size

    
if __name__ == '__main__':
    
    simulator_name = get_script_args(1)[0]
    
    Be=0.1
    Bi=-0.2
    N_rec_v = 10
    
    if '-small' in sys.argv:
        size = 20
        defaultParams.set_total_population_size(20)
        N_rec_v = size
    
    nn_stim_rng = (np.array([0.1, .25, .5, .75, 1])*defaultParams.NI).astype('int')
    nn_stim_rng = (np.array([.75])*defaultParams.NI).astype('int')
    
    if '-small' in sys.argv:
        nn_stim_rng = (np.array([0.5])*defaultParams.NI).astype('int')

    if '-nogui' in sys.argv:
        show_gui = False
    else:
        import matplotlib.pyplot as plt 
        show_gui = True 
        
    
    defaultParams.r_bkg = 9000
    defaultParams.r_stim = 00

    for nn_stim in nn_stim_rng:
        runNetwork(Be, Bi, nn_stim, show_gui=show_gui, save=True,N_rec_v=N_rec_v)
        
    if show_gui:
        plt.show()
