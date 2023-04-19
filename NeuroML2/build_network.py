from neuromllite import Network, Cell, InputSource, Population, Synapse, RectangularRegion
from neuromllite import Projection, RandomConnectivity, Input, Simulation, RandomLayout, OneToOneConnector
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
                   'Be_bkg':defaultParams.Be_bkg,
                   'bkg_rate': 9600}

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
}
    
net.cells.append(cell)


ssp_pre = Cell(id='ssp_pre', 
            pynn_cell='SpikeSourcePoisson',
            parameters = {'rate':'bkg_rate',
                          'start':0,
                          'duration':defaultParams.Ttrans+defaultParams.Tblank})

net.cells.append(ssp_pre)

r = RectangularRegion(id='ISN_network', 
                      x=0,
                      y=0,
                      z=0,
                      width=1000,
                      height=100,
                      depth=1000)
net.regions.append(r)

pE = Population(id='Epop', size='int(N*fraction_E)', component=cell.id, properties={'color':'0 0 0.8'})
pI = Population(id='Ipop', size='N - int(N*fraction_E)', component=cell.id, properties={'color':'.8 0 0'})

bkgPre = Population(id='Background_pre', size='int(N*fraction_E)', component=ssp_pre.id, properties={'color':'.8 .8 .8'})

net.populations.append(pE)
net.populations.append(pI)
net.populations.append(bkgPre)

for p in net.populations:
    p.random_layout = RandomLayout(region=r.id)

net.synapses.append(Synapse(id='ampa', 
                            pynn_receptor_type='excitatory', 
                            pynn_synapse_type='cond_alpha', 
                            parameters={'e_rev':nesp['E_ex'], 'tau_syn':nesp['tau_syn_ex']}))
                            
net.synapses.append(Synapse(id='gaba', 
                            pynn_receptor_type='inhibitory', 
                            pynn_synapse_type='cond_alpha', 
                            parameters={'e_rev':nesp['E_in'], 'tau_syn':nesp['tau_syn_in']}))

net.projections.append(Projection(id='projBkgPre',
                                  presynaptic=bkgPre.id, 
                                  postsynaptic=pE.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight='0.001*Be_bkg',
                                  one_to_one_connector=OneToOneConnector()))
'''
net.projections.append(Projection(id='projEe',
                                  presynaptic=pE.id, 
                                  postsynaptic=pE.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight='0.001*Be',
                                  random_connectivity=RandomConnectivity(probability=0.15)))
                                  
net.projections.append(Projection(id='projEI',
                                  presynaptic=pE.id, 
                                  postsynaptic=pI.id,
                                  synapse='ampa',
                                  delay=2,
                                  weight='0.001*Be',
                                  random_connectivity=RandomConnectivity(probability=0.15)))

net.projections.append(Projection(id='projIE',
                                  presynaptic=pI.id, 
                                  postsynaptic=pE.id,
                                  synapse='gaba',
                                  delay=2,
                                  weight='0.001*Bi',
                                  random_connectivity=RandomConnectivity(probability=1)))

net.projections.append(Projection(id='projII',
                                  presynaptic=pI.id, 
                                  postsynaptic=pI.id,
                                  synapse='gaba',
                                  delay=2,
                                  weight='0.001*Bi',
                                  random_connectivity=RandomConnectivity(probability=1)))
'''

print(net)
print(net.to_json())
new_file = net.to_json_file('%s.json'%net.id)


################################################################################
###   Build Simulation object & save as JSON

sim = Simulation(id='SimISN',
                 network=new_file,
                 duration='1000',
                 dt='0.025',
                 record_traces={pE.id:'*',pI.id:'*'})
                 
sim.to_json_file()


################################################################################
###   Run in some simulators

from neuromllite.NetworkGenerator import check_to_generate_or_run
import sys

check_to_generate_or_run(sys.argv, sim)


