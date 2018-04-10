import numpy as np
import pylab as pl 

# transitent time to discard the data (ms)
Ttrans = 500.
# simulation time before perturbation (ms)
Tblank= 500.
# simulation time of perturbation (ms)
Tstim = 500.
# time after perturbation
Tpost = 500.

T = Ttrans+Tblank+Tstim #+Tpost
N = 500
NE = 400
NI = 100
NI_pert = 95
NI_nonpert = 5

spd = pl.loadtxt('ISN-nest-EI-0.gdf')
spt = spd[:,0]; spi = spd[:,1];

pl.figure()
pl.plot(spt, spi, '|')

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