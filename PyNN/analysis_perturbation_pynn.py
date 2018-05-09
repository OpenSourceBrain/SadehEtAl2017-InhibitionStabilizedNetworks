
from __future__ import print_function
import numpy as np
import pylab as pl 
import matplotlib.patches as patches

import sys
import os
sys.path.append("../SpikingSimulationModels")
import defaultParams
import matplotlib

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

red = '#ff0000'
lblue = '#99c2ff'
lgrey = '#dddddd'
dblue = '#0047b3'
dblue2 = '#444444'

ax1.plot(e_spt, e_spi, '.', color=red,markersize=2)
ax1.plot(pi_spt, pi_spi, '.', color=dblue,markersize=2)
ax1.plot(npi_spt, npi_spi, '.', color=lblue,markersize=2)

ax1.set_ylabel('Cell index')

bw = 100
hst = np.histogram2d(spt, spi, range=((0,T),(0,N-1)), bins=(T/bw,N))

tt = hst[1][0:-1] + np.diff(hst[1])[0]/2
rr = hst[0] / (bw/1000)

r_exc = rr[:,0:NE]
r_inh_pert = rr[:,NE:NE+NI_pert]
r_inh_nonpert = rr[:,N-NI_nonpert:]


r_exc_m = np.nanmean(r_exc,1)
r_inh_pert_m = np.nanmean(r_inh_pert,1)

ks = open('kernelseed')
kernelseed = int(ks.read())
print('kernelseed from last simulation: %i'%kernelseed)
file_rates = 'pertinh.rate.%i.dat'%kernelseed
pertinh = open(file_rates,'w')
print("Saving to file %s list of rates for this seed: %s"%(pertinh,r_inh_pert_m))
for r in r_inh_pert_m:
    pertinh.write('%s\n'%r)
pertinh.close()

all_r = []
# Load all rates files in & average
for f in os.listdir():
    if f.startswith('pertinh.rate'):
        r = pl.loadtxt(f)
        all_r.append(r)
        print("Loaded rates from %s: %s"%(f,r))
        
        #ax2.plot(tt, r, lgrey, lw=0.3)
     
avg_r = []

for i_t in range(len(all_r[0])):
    r = 0
    for i_r in range(len(all_r)):
        r+= all_r[i_r][i_t]
    r/=len(all_r)
    avg_r.append(r)

ax2.plot(tt, avg_r, dblue2, lw=2, linestyle=':')

r_inh_nonpert_m = np.nanmean(r_inh_nonpert,1)

ax2.plot(tt, r_exc_m, red, lw=2)
ax2.plot(tt, r_inh_pert_m, dblue, lw=2)
ax2.plot(tt, r_inh_nonpert_m, lblue, lw=2)



pl.ylabel('Firing rate (Hz)')
pl.xlabel('Time (ms)')

height = ax2.get_ylim()[1]

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

if True:
    matplotlib.pyplot.text(100, 17,'Exc.',  color=red)
    matplotlib.pyplot.text(100, 15, 'Inh. (non pert.)',  color=lblue)
    matplotlib.pyplot.text(100, 13,'Inh. (pert.)',  color=dblue)
    matplotlib.pyplot.text(100, 11,'Inh. (pert.) avg %s'%len(all_r),  color=dblue2)

print('before vs after perturbation')
print('exc: ', np.nanmean(r_exc[t1]), np.nanmean(r_exc[t2]))
print('inh (pert): ', np.nanmean(r_inh_pert_m[t1]), np.nanmean(r_inh_pert_m[t2]))
print('inh (non-pert): ', np.nanmean(r_inh_nonpert_m[t1]), np.nanmean(r_inh_nonpert_m[t2]))

for ax in [ax1,ax2]:    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

fig0.savefig('rates.png', dpi=150, bbox_inches='tight')

if not '-nogui' in sys.argv:
    pl.show()