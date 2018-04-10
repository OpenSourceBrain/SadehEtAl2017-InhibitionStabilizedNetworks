
from __future__ import print_function
import numpy as np
import pylab as pl 

import sys
sys.path.append("../SpikingSimulationModels")
import defaultParams

# transitent time to discard the data (ms)
Ttrans = defaultParams.Ttrans
# simulation time before perturbation (ms)
Tblank= defaultParams.Tblank
# simulation time of perturbation (ms)
Tstim = defaultParams.Tstim
# time after perturbation
Tpost =  0 # 500.

T = Ttrans+Tblank+Tstim #+Tpost
N = defaultParams.N
NE = defaultParams.NE
NI = defaultParams.NI

fraction_to_stim = 0.75
if len(sys.argv)==2:
    try:
        fraction_to_stim = float(sys.argv[1])
    except:
        pass

NI_pert = int(NI * fraction_to_stim)
NI_nonpert = NI-NI_pert

spd = pl.loadtxt('ISN-nest-EI-0.gdf')
spt = spd[:,0]; spi = spd[:,1];


e_spt = []
e_spi = []
pi_spt = []
pi_spi = []
npi_spt = []
npi_spi = []

for k in range(len(spt)):
    t = spt[k]
    i = spi[k]
    if i<NE:
        e_spi.append(i)
        e_spt.append(t)
    elif i<NE+NI_pert:
        pi_spi.append(i)
        pi_spt.append(t)
    else:
        npi_spi.append(i)
        npi_spt.append(t)
        
pl.figure()

pl.plot(e_spt, e_spi, '|', color='red')
pl.plot(pi_spt, pi_spi, '|', color='black')
pl.plot(npi_spt, npi_spi, '|', color='blue')

bw = 50
hst = np.histogram2d(spt, spi, range=((0,T),(0,N-1)), bins=(T/bw,N))

tt = hst[1][0:-1] + np.diff(hst[1])[0]/2
rr = hst[0] / (bw/1000)

r_exc = rr[:,0:NE]
r_inh_pert = rr[:,NE:NE+NI_pert]
r_inh_nonpert = rr[:,N-NI_nonpert:]

pl.figure()

r_exc_m = np.nanmean(r_exc,1)
r_inh_pert_m = np.nanmean(r_inh_pert,1)
r_inh_nonpert_m = np.nanmean(r_inh_nonpert,1)

pl.plot(tt, r_exc_m, 'r', lw=2)
pl.plot(tt, r_inh_pert_m, 'k', lw=2)
pl.plot(tt, r_inh_nonpert_m, 'b', lw=2)

t1 = (tt>Ttrans)*(tt<(Ttrans+Tblank))
t2 = (tt>(Ttrans+Tblank))*(tt<(Ttrans+Tblank+Tstim))

print('before vs after perturbation')
print('exc: ', np.nanmean(r_exc[t1]), np.nanmean(r_exc[t2]))
print('inh (pert): ', np.nanmean(r_inh_pert_m[t1]), np.nanmean(r_inh_pert_m[t2]))
print('inh (non-pert): ', np.nanmean(r_inh_nonpert_m[t1]), np.nanmean(r_inh_nonpert_m[t2]))

pl.show()