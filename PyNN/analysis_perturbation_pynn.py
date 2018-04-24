
from __future__ import print_function
import numpy as np
import pylab as pl 
import matplotlib.patches as patches

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
Tpost =  defaultParams.Tpost

T = Ttrans+Tblank+Tstim+Tpost


fraction_to_stim = 0.75

if len(sys.argv)>=2:
    try:
        fraction_to_stim = float(sys.argv[1])
    except:
        pass
    
if len(sys.argv)>=3:
    try:
        size = int(sys.argv[2])
        defaultParams.set_total_population_size(size)
    except:
        pass


N = defaultParams.N
NE = defaultParams.NE
NI = defaultParams.NI

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
        
fig0, (ax1, ax2) = pl.subplots(2, 1, sharex=True)

fig0.set_size_inches(12, 8, forward=True)

lblue = '#99c2ff'
dblue = '#0047b3'

ax1.plot(e_spt, e_spi, '.', color='red',markersize=2)
ax1.plot(pi_spt, pi_spi, '.', color=dblue,markersize=2)
ax1.plot(npi_spt, npi_spi, '.', color=lblue,markersize=2)

ax1.set_ylabel('Cell index')

bw = 50
hst = np.histogram2d(spt, spi, range=((0,T),(0,N-1)), bins=(T/bw,N))

tt = hst[1][0:-1] + np.diff(hst[1])[0]/2
rr = hst[0] / (bw/1000)

r_exc = rr[:,0:NE]
r_inh_pert = rr[:,NE:NE+NI_pert]
r_inh_nonpert = rr[:,N-NI_nonpert:]


r_exc_m = np.nanmean(r_exc,1)
r_inh_pert_m = np.nanmean(r_inh_pert,1)
r_inh_nonpert_m = np.nanmean(r_inh_nonpert,1)

ax2.plot(tt, r_exc_m, 'r', lw=2)
ax2.plot(tt, r_inh_pert_m, dblue, lw=2)
ax2.plot(tt, r_inh_nonpert_m, lblue, lw=2)

pl.ylabel('Firing rate (Hz)')
pl.xlabel('Time (ms)')

height = 20

pl.xlim([0,T])
pl.ylim([0,height])

ax2.add_patch(
    patches.Rectangle(
        (Ttrans+Tblank, 0),   # (x,y)
        Tstim,          # width
        height,          # height
        facecolor='#f8f5dd'
    )
)

t1 = (tt>Ttrans)*(tt<(Ttrans+Tblank))
t2 = (tt>(Ttrans+Tblank))*(tt<(Ttrans+Tblank+Tstim))

print('before vs after perturbation')
print('exc: ', np.nanmean(r_exc[t1]), np.nanmean(r_exc[t2]))
print('inh (pert): ', np.nanmean(r_inh_pert_m[t1]), np.nanmean(r_inh_pert_m[t2]))
print('inh (non-pert): ', np.nanmean(r_inh_nonpert_m[t1]), np.nanmean(r_inh_nonpert_m[t2]))

fig0.savefig('rates.png', dpi=150, bbox_inches='tight')

pl.show()