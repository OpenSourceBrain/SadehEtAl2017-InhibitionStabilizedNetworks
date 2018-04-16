from neuromllite import Network, Cell, InputSource, Population, Synapse
from neuromllite import Projection, RandomConnectivity, Input, Simulation, RandomLayout
from neuromllite.NetworkGenerator import generate_and_run
from neuromllite.NetworkGenerator import generate_neuroml2_from_network
import sys

sys.path.append("../SpikingSimulationModels")
import defaultParams

################################################################################
###   Build new network

net = Network(id='ISN')
net.notes = 'Based on network of Sadeh et al. 2017'

net.parameters = { 'N': 10, 
                   'fraction_E': 0.8,
                   'fraction_stimulated': .75,
                   'Be': .4,
                   'Bi': .5,
                   'bkg_rate': 960}

cell = Cell(id='eifcell', pynn_cell='EIF_cond_alpha_isfa_ista')

nesp = defaultParams.neuron_params_default
cell.parameters = {
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
    
net.cells.append(cell)


ssp_pre = Cell(id='ssp_pre', 
            pynn_cell='SpikeSourcePoisson',
            parameters = {'rate':'bkg_rate',
                          'start':0,
                          'duration':defaultParams.Ttrans+defaultParams.Tblank})

net.cells.append(ssp_pre)


pE = Population(id='Epop', size='int(N*fraction_E)', component=cell.id, properties={'color':'0 0 0.8'})
pI = Population(id='Ipop', size='N - int(N*fraction_E)', component=cell.id, properties={'color':'.8 0 0'})

bkgPre = Population(id='Background_pre', size='N', component=ssp_pre.id, properties={'color':'.8 .8 .8'})

net.populations.append(pE)
net.populations.append(pI)
net.populations.append(bkgPre)

for p in net.populations:
    p.random_layout = RandomLayout(width=1000,height=100,depth=1000)

net.synapses.append(Synapse(id='ampa', 
                            pynn_receptor_type='excitatory', 
                            pynn_synapse_type='cond_alpha', 
                            parameters={'e_rev':-10, 'tau_syn':2}))
net.synapses.append(Synapse(id='gaba', 
                            pynn_receptor_type='inhibitory', 
                            pynn_synapse_type='cond_alpha', 
                            parameters={'e_rev':-80, 'tau_syn':10}))

net.projections.append(Projection(id='projBkgPre',
                                  presynaptic=bkgPre.id, 
                                  postsynaptic=pE.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight=0.02,
                                  random_connectivity=RandomConnectivity(probability=1)))
'''
net.projections.append(Projection(id='projEe',
                                  presynaptic=pE.id, 
                                  postsynaptic=pE.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight=0.02,
                                  random_connectivity=RandomConnectivity(probability=.05)))
                                  
net.projections.append(Projection(id='projEI',
                                  presynaptic=pE.id, 
                                  postsynaptic=pI.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight=0.02,
                                  random_connectivity=RandomConnectivity(probability=.05)))

net.projections.append(Projection(id='projIE',
                                  presynaptic=pI.id, 
                                  postsynaptic=pE.id,
                                  synapse='gaba',
                                  delay=2,
                                  weight=0.02,
                                  random_connectivity=RandomConnectivity(probability=.05)))'''


print(net)
print(net.to_json())
net.to_json_file('%s.json'%net.id)


################################################################################
###   Build Simulation object & save as JSON

sim = Simulation(id='SimISN',
                 duration='1000',
                 dt='0.025',
                 recordTraces={pE.id:'*',pI.id:'*'})
                 
sim.to_json_file()

if '-pynnnest' in sys.argv:
    generate_and_run(sim, net, simulator='PyNN_NEST')
    
elif '-pynnnrn' in sys.argv:
    generate_and_run(sim, net, simulator='PyNN_NEURON')
    
elif '-pynnbrian' in sys.argv:
    generate_and_run(sim, net, simulator='PyNN_Brian')
    
elif '-netpyne' in sys.argv:
    generate_and_run(sim, net, simulator='NetPyNE')
    
elif '-jnmlnrn' in sys.argv:
    generate_and_run(sim, net, simulator='jNeuroML_NEURON')
    
elif '-jnmlnetpyne' in sys.argv:
    generate_and_run(sim, net, simulator='jNeuroML_NetPyNE')
    
elif '-sonata' in sys.argv:
    generate_and_run(sim, net, simulator='Sonata') # Will not "run" obviously...
    
elif '-graph' in sys.argv:
    generate_and_run(sim, net, simulator='Graph') # Will not "run" obviously...
    
elif '-jnml' in sys.argv:
    generate_and_run(sim, net, simulator='jNeuroML')


