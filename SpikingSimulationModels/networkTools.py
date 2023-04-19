
################################################################################
# -- Tools for simulating networks of spiking neurons using the NEST simulator
################################################################################

import numpy as np; import pylab as pl; import time, os, sys
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *;
import nest

# --- NEST initialization
def _nest_start_(n_cores=n_cores):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt,
                        "print_time": True,
                        "overwrite_files": True,
                        'local_num_threads': n_cores})
    nest.set_verbosity("M_WARNING")

# --- Get simulation time of the network
def _simulation_time_():
    return nest.GetStatus([0])[0]['time']

# --- Define nest-compatible "Connect"
bNestUseConvergentConnect = "ConvergentConnect" in dir(nest)

def ConvConnect(arg0, arg1, syn_spec='static_synapse', *args):
    return nest.Connect(arg0, arg1, syn_spec=syn_spec)

def DivConnect(arg0, arg1, syn_spec='static_synapse', *args):
    return nest.Connect(arg0, arg1, syn_spec=syn_spec)

# --- Making neurons
def _make_neurons_(N, neuron_model="iaf_cond_alpha", myparams={}):
    nest.SetDefaults(neuron_model, neuron_params_default)
    neurons = nest.Create(neuron_model, N)
    if myparams != {}:
        for nn in range(N):
            for kk in myparams.keys():
                #nest.SetStatus([neurons[nn]], {kk:myparams[kk][nn]})
                #print("Setting %s to %s on neuron %s"%(kk, myparams[kk][nn], nn))
                nest.SetStatus(neurons[nn], {kk:myparams[kk][nn]})
    return neurons

# --- Generating (poisson) inputs and setting their firing rates
def _poisson_inp_(N):
    poisson_inp = nest.Create("poisson_generator", N)
    return poisson_inp

def _set_rate_(neurons, rates):
    for ii, nn in enumerate(neurons):
        nest.SetStatus([nn], {'rate':rates[ii]})

# --- Setting the rates of background inputs and conencting them
def _bkg_inp_(neurons, bkg_rates, bkg_w = Be*1e9):
    N = len(neurons)
    bkg_inp = nest.Create("poisson_generator", N)
    for ii in range(N):
        nest.SetStatus([bkg_inp[ii]], {'rate':bkg_rates[ii]})
    for ii in range(N):
        nest.Connect([bkg_inp[ii]], [neurons[ii]], \
        syn_spec = {'weight':bkg_w, 'delay':delay_default})

# --- Copy to a parrot neuron (mirroring every spike)
def _copy_to_parrots_(pre_pop):
    parrots = nest.Create('parrot_neuron', len(pre_pop))
    nest.Connect(pre_pop, parrots, conn_spec='one_to_one')
    return parrots

# --- Defining synapse types
def _define_synapse_(syn_type='static_synapse', name='exc', w=Be*1e9, d=delay_default):
    nest.CopyModel(syn_type, name, {"weight":w, "delay":d})

# --- Recording and reading spikes and voltages
def _recording_spikes_(neurons, start=0., stop=np.inf, to_file=False, to_memory=True):
    spikes = nest.Create("spike_recorder", 1)
    nest.SetStatus(spikes, {"label":'spike-det',
                        "start": start,
                        "stop": stop})

    ConvConnect(neurons, spikes)
    return spikes

def _recording_voltages_(neurons, start=0., stop=np.inf):
    voltages = nest.Create("voltmeter")
    nest.SetStatus(voltages, {"label":'volt-meter',
                        "start": start,
                        "stop": stop})
    DivConnect(voltages, neurons)
    return voltages

def _reading_spikes_(spikes):
    spike_data = nest.GetStatus(spikes)[0]['events']
    return spike_data

def _reading_voltages_(voltages):
    voltage_data = nest.GetStatus(voltages)[0]['events']
    return voltage_data

# --- Connect population A to population B with weight matrix W
def _connect_pops_(pre_pop, post_pop, weight, syn_model='static'):
    print(weight.shape)
    dd = dt + delay_default*np.ones(weight.T.shape)
    nest.Connect(pre_pop, post_pop, syn_spec = {'weight':weight.T, 'delay':dd})
    
    #for ii, nn in enumerate(pre_pop):
    #    ww = weight[ii]
    #    dd = dt + delay_default*np.ones(len(ww))
    #    nest.Connect([nn], post_pop, syn_spec = {'weight':ww.tolist(), 'delay':dd.tolist()})
    print("  Created %s non zero weight connections from %s to %s, %s"%(np.count_nonzero(weight),_pop_info_(pre_pop), _pop_info_(post_pop), syn_model))
    
def _pop_info_(pop):
    return 'Cells %s->%s (%s total)' %(pop[0],pop[-1],len(pop))

# --- Run the simulation for the duration of T (ms)
def _run_simulation_(T):
    nest.Simulate(T)

################################################################################
################################################################################
################################################################################
