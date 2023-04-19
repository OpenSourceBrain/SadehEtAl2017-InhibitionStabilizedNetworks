
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
    ker = norm.pdf(np.linspace(-int(ker_size/2),int(ker_size/2),int(ker_size)), scale=ker_size/5)
    zz = np.convolve(xx, ker, 'size')
    return zz

def analyse(fraction_to_stim, size, 
            file_to_save = 'rates.png', 
            show=True, 
            detailed_also=False, 
            average_prev_runs=False,
            legend=False):
    
    print("Will plot average of previous runs: %s"%average_prev_runs)
    # transient time to discard the data (ms)
    Ttrans = defaultParams.Ttrans
    # simulation time before perturbation (ms)
    Tblank= defaultParams.Tblank
    # simulation time of perturbation (ms)
    Tstim = defaultParams.Tstim
    # time after perturbation
    Tpost =  defaultParams.Tpost

    T = Ttrans+Tblank+Tstim+Tpost

    defaultParams.set_total_population_size(size)

    N = defaultParams.N
    NE = defaultParams.NE  # num exc point neurons
    NE2 = 0 # num detailed cells
    
    if detailed_also:
        NE2 = 10
        NE=NE-NE2
    
    NI = defaultParams.NI  # num inh neurons

    NI_pert = int(NI * fraction_to_stim)
    NI_nonpert = NI-NI_pert

    spd = pl.loadtxt('ISN-nest-EI-0.gdf')
    spt = spd[:,0]; spi = spd[:,1];


    e_spt = []
    e_spi = []
    e2_spt = []
    e2_spi = []
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
        elif i<NE+NE2:
            e2_spi.append(i)
            e2_spt.append(t)
        elif i<NE+NE2+NI_pert:
            pi_spi.append(i)
            pi_spt.append(t)
        else:
            npi_spi.append(i)
            npi_spt.append(t)

    if not detailed_also:
        fig0, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
    else:
        fig0, (ax1, ax1b, ax2) = pl.subplots(3, 1, sharex=True, gridspec_kw = {'height_ratios':[3, 1, 4]})

    fig0.set_size_inches(12, 8, forward=True)

    red = '#ff0000'
    lblue = '#99c2ff'
    lgrey = '#dddddd'
    green = '#00cc00'
    dgreen = '#008800'
    dblue = '#0047b3'
    dblue2 = '#444444'

    ax1.plot(pi_spt, pi_spi, '.', color=dblue,markersize=2)
    ax1.plot(npi_spt, npi_spi, '.', color=lblue,markersize=2)
    if NE2>0:
        ax1.plot(e2_spt, e2_spi, '.', color=green,markersize=7)
    ax1.plot(e_spt, e_spi, '.', color=red,markersize=2)
    
    if NE2>0:
        ax1b.plot(e2_spt, e2_spi, '.', color=green,markersize=5)
        ax1b.set_ylabel('Cell index')

    ax1.set_ylabel('Cell index')

    yl = ax1.get_ylim()

    ax1.set_ylim([yl[1],yl[0]])
    
    
    if NE2>0:
        ylb = ax1b.get_ylim()

        #ax1b.set_ylim([ylb[1]-1,ylb[0]+1])
        ax1b.set_yticks([NE,NE+NE2-1])

    bw = 1#0
    hst = np.histogram2d(spt, spi, range=((0,T),(0,N-1)), bins=(int(T/bw),N))

    tt = hst[1][0:-1] + np.diff(hst[1])[0]/2
    rr = hst[0] / (bw/1000)

    r_exc = rr[:,0:NE]
    r_exc2 = rr[:,NE:NE+NE2]
    r_inh_pert = rr[:,NE+NE2:NE+NE2+NI_pert]
    r_inh_nonpert = rr[:,N-NI_nonpert:]

    sm_len = 151.0
    r_exc_m = mySmooth(np.nanmean(r_exc,1),sm_len)[0:rr.shape[0]]
    r_exc2_m = mySmooth(np.nanmean(r_exc2,1),sm_len)[0:rr.shape[0]]
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
    
    if detailed_also:
        file_rates2 = 'exc2.rate.%i.dat'%kernelseed
        exc2 = open(file_rates2,'w')
        print("Saving to file %s list of rates for this seed: %s"%(exc2,r_exc2_m))
        for r in r_exc2_m:
            exc2.write('%s\n'%r)
        exc2.close()

    all_ri = []
    all_re2 = []
    
    if average_prev_runs:
        # Load all rates files in & average
        for f in os.listdir('.'):
            if f.startswith('pertinh.rate'):
                r = pl.loadtxt(f)
                all_ri.append(r)
                print("Loaded rates from %s: %s"%(f,r))
            if f.startswith('exc2.rate'):
                r = pl.loadtxt(f)
                all_re2.append(r)
                print("Loaded rates from %s: %s"%(f,r))

                #ax2.plot(tt, r, lgrey, lw=0.3)

    avg_ri = []
    avg_re2 = []

    if len(all_ri)>0:
        for i_t in range(len(all_ri[0])):
            r = 0
            for i_r in range(len(all_ri)):
                r+= all_ri[i_r][i_t]
            r/=len(all_ri)
            avg_ri.append(r)
    if len(all_re2)>0:
        for i_t in range(len(all_re2[0])):
            r = 0
            for i_r in range(len(all_re2)):
                r+= all_re2[i_r][i_t]
            r/=len(all_re2)
            avg_re2.append(r)

    if len(all_ri)>0:
        ax2.plot(tt, avg_ri, dblue2, lw=4, linestyle=':')
        
    if NE2>0:
        ax2.plot(tt, avg_re2, dgreen, lw=4, linestyle=':')
    
    
    

    ax2.plot(tt, r_exc_m, red, lw=2)
    if NE2>0:
        ax2.plot(tt, r_exc2_m, green, lw=2)
    ax2.plot(tt, r_inh_pert_m, dblue, lw=2)
    ax2.plot(tt, r_inh_nonpert_m, lblue, lw=2)

    pl.ylabel('Firing rate (Hz)')
    pl.xlabel('Time (ms)')

    height = ax2.get_ylim()[1]
    
    if detailed_also: height = 12

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

    if legend:
        j = 3
        matplotlib.pyplot.text(100, j+17,'Exc.',  color=red)
        i=0
        if detailed_also:
            matplotlib.pyplot.text(100, j+15,'Detailed Exc.',  color=green)
            matplotlib.pyplot.text(100, j+13,'Detailed Exc. avg %s'%len(all_re2),  color=dgreen)
            i=-4
            
        matplotlib.pyplot.text(100, j+15+i, 'Inh. (non pert.)',  color=lblue)
        matplotlib.pyplot.text(100, j+13+i,'Inh. (pert.)',  color=dblue)
        matplotlib.pyplot.text(100, j+11+i,'Inh. (pert.) avg %s'%len(all_ri),  color=dblue2)

    print('before vs after perturbation')
    print('exc: ', np.nanmean(r_exc[t1]), np.nanmean(r_exc[t2]))
    print('inh (pert): ', np.nanmean(r_inh_pert_m[t1]), np.nanmean(r_inh_pert_m[t2]))
    print('inh (non-pert): ', np.nanmean(r_inh_nonpert_m[t1]), np.nanmean(r_inh_nonpert_m[t2]))

    all_ax = [ax1,ax2]
    if detailed_also: all_ax.append(ax1b)
    for ax in all_ax:    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    fig0.savefig(file_to_save, dpi=150, bbox_inches='tight')

    if show:
        pl.show()


if __name__ == '__main__':
    
    if not sys.version_info[0] == 3:
        print('Please run with Python 3...')
        quit()
        
    show = not '-nogui' in sys.argv
    
    
    detailed_also = '-detailed' in sys.argv
    average_prev_runs = '-average' in sys.argv
    legend = '-legend' in sys.argv
    
    fraction_to_stim = 0.75

    if len(sys.argv)>=2:
        try:
            fraction_to_stim = float(sys.argv[1])
        except:
            pass

    if len(sys.argv)>=3:
            size = int(sys.argv[2])
        
    analyse(fraction_to_stim, size, file_to_save='rates.png',show=show, detailed_also=detailed_also, average_prev_runs=average_prev_runs, legend=legend)
        
