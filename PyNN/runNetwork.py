
from __future__ import print_function
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
               simtime = defaultParams.Tpost+defaultParams.Tstim+defaultParams.Tblank+defaultParams.Ttrans, 
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

    print('\n # -----> Num cells: %s, size of pert. inh: %s; base rate %s; pert rate %s'% (N, nn_stim, defaultParams.r_bkg, defaultParams.r_stim))

    r_extra = np.zeros(N)
    r_extra[NE:NE+nn_stim] = defaultParams.r_stim

    rr1 = defaultParams.r_bkg*np.random.uniform(.75,1.25, N)
    rr2 = rr1 + r_extra
    
    rank = setup(timestep=dt, max_delay=defaultParams.delay_default, reference='ISN', **extra)
    
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
    
    
    nesp = defaultParams.neuron_params_default
    cell_parameters = {
        'cm':         nesp['C_m']/1000,   # Capacitance of the membrane in nF
        'tau_refrac': nesp['t_ref'],     # Duration of refractory period in ms.
        'v_spike':    0.0 ,     # Spike detection threshold in mV.   https://github.com/nest/nest-simulator/blob/master/models/aeif_cond_alpha.cpp
        'v_reset':    nesp['V_reset'],     # Reset value for V_m after a spike. In mV.
        'v_rest':     nesp['E_L'],     # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m':      nesp['C_m']/nesp['g_L'],  # Membrane time constant in ms = cm/tau_m*1000.0, C_m/g_L
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

    print("%d Creating population with %d neurons." % (rank, N))
    celltype = EIF_cond_alpha_isfa_ista(**cell_parameters)
    celltype.default_initial_values['v'] = cell_parameters['v_rest'] # Setting default init v, useful for NML2 export
    EI_pop = Population(N, celltype, label="EI")
    E_pop = PopulationView(EI_pop, np.array(range(0,NE)),label='E_pop')
    #print("%d Creating pop view %s." % (rank, E_pop))
    I_pop = PopulationView(EI_pop, np.array(range(NE,N)),label='I_pop')
    #print("%d Creating pop view %s." % (rank, I_pop))
    
    I_pert_pop = PopulationView(EI_pop, np.array(range(NE,NE+nn_stim)),label='I_pert_pop')
    I_nonpert_pop = PopulationView(EI_pop, np.array(range(0,NE)+range(NE+nn_stim,N)),label='I_nonpert_pop')
    
    p_rate = defaultParams.r_bkg
    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_typeA = SpikeSourcePoisson(rate=p_rate, start=0,duration=defaultParams.Ttrans+defaultParams.Tblank)
    expoissonA = Population(N, source_typeA, label="pre_pert_stim_all")
    
    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_typeB = SpikeSourcePoisson(rate=p_rate, start=defaultParams.Ttrans+defaultParams.Tblank)
    expoissonB = Population(N-nn_stim, source_typeB, label="non_pert_stim")
    
    p_rate = defaultParams.r_bkg+defaultParams.r_stim
    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_typeC = SpikeSourcePoisson(rate=p_rate, start=defaultParams.Ttrans+defaultParams.Tblank, duration=defaultParams.Tstim)
    expoissonC = Population(nn_stim, source_typeC, label="pert_stim")

    p_rate = defaultParams.r_bkg
    print("%d Creating excitatory Poisson generator with rate %g spikes/s." % (rank, p_rate))
    source_typeD = SpikeSourcePoisson(rate=p_rate, start=defaultParams.Ttrans+defaultParams.Tblank+defaultParams.Tstim)
    expoissonD = Population(nn_stim, source_typeD, label="pert_poststim")
    

    progress_bar = ProgressBar(width=20)
    connector_E = FixedProbabilityConnector(0.15, rng=rng, callback=progress_bar)
    connector_I = FixedProbabilityConnector(1, rng=rng, callback=progress_bar)
    
    EE_syn = StaticSynapse(weight=0.001*Bee, delay=defaultParams.delay_default)
    EI_syn = StaticSynapse(weight=0.001*Bei, delay=defaultParams.delay_default)
    II_syn = StaticSynapse(weight=0.001*Bii, delay=defaultParams.delay_default)
    IE_syn = StaticSynapse(weight=0.001*Bie, delay=defaultParams.delay_default)
    
    #I_syn = StaticSynapse(weight=JI, delay=delay)
    ext_Connector = OneToOneConnector(callback=progress_bar)
    ext_syn_bkg = StaticSynapse(weight=0.001*defaultParams.Be_bkg, delay=defaultParams.delay_default)
    ext_syn_stim = StaticSynapse(weight=0.001*defaultParams.Be_stim, delay=defaultParams.delay_default)
    
    
    E_to_E = Projection(E_pop, E_pop, connector_E, EE_syn, receptor_type="excitatory")
    print("E --> E\t\t", len(E_to_E), "connections")
    E_to_I = Projection(E_pop, I_pop, connector_E, EI_syn, receptor_type="excitatory")
    print("E --> I\t\t", len(E_to_I), "connections")
    I_to_I = Projection(I_pop, I_pop, connector_I, II_syn, receptor_type="inhibitory")
    print("I --> I\t\t", len(I_to_I), "connections")
    I_to_E = Projection(I_pop, E_pop, connector_I, IE_syn, receptor_type="inhibitory")
    print("I --> E\t\t", len(I_to_E), "connections")
    
    
    input_A = Projection(expoissonA, EI_pop, ext_Connector, ext_syn_bkg, receptor_type="excitatory")
    print("input --> %s cells pre pert\t"%len(EI_pop), len(input_A), "connections")
    input_B = Projection(expoissonB, I_nonpert_pop, ext_Connector, ext_syn_bkg, receptor_type="excitatory")
    print("input --> %s cells post pert\t"%len(I_nonpert_pop), len(input_B), "connections")
    input_C = Projection(expoissonC, I_pert_pop, ext_Connector, ext_syn_stim, receptor_type="excitatory")
    print("input --> %s cells pre pert\t"%len(I_pert_pop), len(input_C), "connections")
    input_D = Projection(expoissonD, I_pert_pop, ext_Connector, ext_syn_stim, receptor_type="excitatory")
    print("input --> %s cells pre pert\t"%len(I_pert_pop), len(input_D), "connections")
    
    
    # Record spikes
    print("%d Setting up recording in excitatory population." % rank)
    EI_pop.record('spikes')
    if N_rec_v>0:
        EI_pop[0:min(N,N_rec_v)].record('v')
    
    
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
        for pop in [EI_pop]:
            io = PyNNTextIO(filename="ISN-%s-%s-%i.gdf"%(simulator_name, pop.label, rank))
            spikes =  pop.get_data('spikes', gather=False)
            for segment in spikes.segments:
                io.write_segment(segment)
                
            vs =  pop.get_data('v', gather=False)
            for segment in vs.segments:
                for i in range(len(segment.analogsignals[0].transpose())):
                    filename="ISN-%s-%s-cell%i.dat"%(simulator_name, pop.label, i)
                    vm = segment.analogsignals[0].transpose()[i]
                    tt = np.array([t*dt/1000. for t in range(len(vm))])
                    times_vm = np.array([tt, vm/1000.]).transpose()
                    np.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')
            
    spike_data = {}
    spike_data['senders'] = []
    spike_data['times'] = []
    index_offset = 1
    for pop in [EI_pop]:
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


    print("Build time         : %g s" % buildCPUTime)
    print("Simulation time    : %g s" % simCPUTime)

    # === Clean up and quit ========================================================

    end()
    
if __name__ == '__main__':
    
    simulator_name = get_script_args(1)[0]
    
    Be=.4
    Bi=.5
    N_rec_v = 20
    
    
    fraction_to_stim = 0.75
    
    if len(sys.argv)>=3:
        try:
            fraction_to_stim = float(sys.argv[2])
        except:
            pass
        
    if len(sys.argv)>=4:
        try:
            size = int(sys.argv[3])
            defaultParams.set_total_population_size(size)
        except:
            pass
        
    print("Going to stimulate %s of the inhibitory cells"%fraction_to_stim)
    nn_stim_rng = (np.array([fraction_to_stim])*defaultParams.NI).astype('int')
    
    '''
    if '-small' in sys.argv:
        nn_stim_rng = (np.array([0.95])*defaultParams.NI).astype('int')
        defaultParams.r_bkg = 9.6e3
        defaultParams.r_stim = -100'''

    if '-nogui' in sys.argv:
        show_gui = False
    else:
        import matplotlib.pyplot as plt 
        show_gui = True 
        

    for nn_stim in nn_stim_rng:
        runNetwork(Be, Bi, nn_stim, show_gui=show_gui, save=True,N_rec_v=N_rec_v)
        
    if show_gui:
        plt.show()
        
