
from __future__ import print_function
import numpy as np
import pylab as pl 
import matplotlib.patches as patches
from scipy.stats import norm

import sys
import os
sys.path.append("../SpikingSimulationModels")
import defaultParams
import matplotlib

def movingAverage(xx, N=1):
    y, moving_aves = [0], np.zeros(len(xx))
    for i, x in enumerate(xx):
        y.append(y[i-1] + x)
        if i>=N:
            moving_aves[i] = (y[i] - y[i-N])/N
        else:
            moving_aves[i] = np.nan;
    return moving_aves

def smooth(x,window_len=11,window='flat'):#hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def mySmooth(xx, ker_size = 51):
    ker = norm.pdf(np.linspace(-int(ker_size/2),int(ker_size/2),ker_size), scale=ker_size/5)
    zz = np.convolve(xx, ker, 'size')
    return zz

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

ax1.plot(pi_spt, pi_spi, '.', color=dblue,markersize=2)
ax1.plot(npi_spt, npi_spi, '.', color=lblue,markersize=2)
ax1.plot(e_spt, e_spi, '.', color=red,markersize=2)

ax1.set_ylabel('Cell index')

yl = ax1.get_ylim()

ax1.set_ylim([yl[1],yl[0]])

bw = 1#0
hst = np.histogram2d(spt, spi, range=((0,T),(0,N-1)), bins=(T/bw,N))

tt = hst[1][0:-1] + np.diff(hst[1])[0]/2
rr = hst[0] / (bw/1000)

r_exc = rr[:,0:NE]
r_inh_pert = rr[:,NE:NE+NI_pert]
r_inh_nonpert = rr[:,N-NI_nonpert:]

sm_len = 101
r_exc_m = mySmooth(np.nanmean(r_exc,1),sm_len)[0:rr.shape[0]]
r_inh_pert_m = mySmooth(np.nanmean(r_inh_pert,1),sm_len)[0:rr.shape[0]]
r_inh_nonpert_m = mySmooth(np.nanmean(r_inh_nonpert,1),sm_len)[0:rr.shape[0]]

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

ax2.plot(tt, avg_r, dblue2, lw=4, linestyle=':')

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
