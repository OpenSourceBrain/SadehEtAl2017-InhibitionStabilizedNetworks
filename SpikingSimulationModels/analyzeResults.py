
################################################################################
# -- Preprocessing and analysis of the simulation results
################################################################################

import numpy as np; import pylab as pl; import os, pickle
from scipy.interpolate import interp1d
from imp import reload
import defaultParams; reload(defaultParams); from defaultParams import *

cwd = os.getcwd()

nns_finer = np.arange(nn_stim_rng[0], nn_stim_rng[-1], 1)

# -- compute mean differential rates & the critical size of inh pert
rdIp_all = {}
zip_finer_all = {}
nn_crit_all = np.zeros((len(Be_rng), len(Bi_rng)))
for ij1, Be in enumerate(Be_rng):
    for ij2, Bi in enumerate(Bi_rng):

        os.chdir(cwd+'/SimulationResults')
        sim_name = 'sim_res_Be'+str(Be)+'_Bi'+str(Bi)
        fl = open(sim_name, 'rb'); sim_res = pickle.load(fl); fl.close()

        rd_E = np.zeros((len(nn_stim_rng), Ntrials))
        rd_Ip, rd_Inp = np.zeros(rd_E.shape), np.zeros(rd_E.shape)
        for ii, nn_stim in enumerate(nn_stim_rng):
            rd_E[ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,0:NE],1),1) \
                     - np.mean(np.mean(sim_res[nn_stim][0][:,:,0:NE],1),1)

            rd_Ip[ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,NE:NE+nn_stim],1),1) \
                      - np.mean(np.mean(sim_res[nn_stim][0][:,:,NE:NE+nn_stim],1),1)

            rd_Inp[ii] = np.mean(np.mean(sim_res[nn_stim][1][:,:,NE+nn_stim:],1),1) \
                        - np.mean(np.mean(sim_res[nn_stim][0][:,:,NE+nn_stim:],1),1)

        zip_func = interp1d(nn_stim_rng, np.nanmean(rd_Ip,1), 'linear')
        zip_finer = zip_func(nns_finer)

        zzz = (zip_finer<.1)[0:-1] * (zip_finer>0)[0:-1] * (np.diff(zip_finer)>0)
        if np.sum(zzz) > 0:
            nn_crit = nns_finer[int(np.nanmean(np.where(zzz == 1)[0]))]
        else: nn_crit = np.nan

        zip_finer_all[Be,Bi] = zip_finer
        nn_crit_all[ij1,ij2] = nn_crit
        rdIp_all[Be, Bi] = rd_Ip

analysis_results = [rdIp_all, nns_finer, zip_finer_all,  nn_crit_all]

fl = open('analysis_results', 'wb'); pickle.dump(analysis_results, fl); fl.close()

os.chdir(cwd)

################################################################################
################################################################################
################################################################################
