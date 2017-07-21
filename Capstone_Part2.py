# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure is an example from astroML: see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_imaging_sample

from matplotlib.patches import Ellipse
from scipy.stats import norm

from sklearn.cluster import KMeans
from sklearn import preprocessing

from astroML.datasets import fetch_sdss_sspp

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
from astroML.datasets import fetch_imaging_sample
setup_text_plots(fontsize=8, usetex=True)
#comment out later
ax1 = ax2 = ax3 = ax4 = "hi"
#------------------------------------------------------------
# Get the star/galaxy data
data = fetch_imaging_sample()

objtype = data['type']

stars = data[objtype == 6][:5000]
galaxies = data[objtype == 3][:5000]
X = np.vstack([galaxies['gRaw'], galaxies['rRaw']]).T


H, FeH_bins, alphFe_bins = np.histogram2d(galaxies['gRaw'], galaxies['rRaw'], 50)

n_clusters = 2

scaler = preprocessing.StandardScaler()
clf = KMeans(n_clusters)
clf.fit(scaler.fit_transform(X))

#------------------------------------------------------------
# Plot the stars and galaxies
plot_kwargs = dict(color='k', linestyle='none', marker='.', markersize=1)

fig = plt.figure()
ax2 = fig.add_subplot(221)
ax2.plot(galaxies['gRaw'] - galaxies['rRaw'],
         galaxies['rRaw'] - galaxies['iRaw'],
         **plot_kwargs)
ax1 = fig.add_subplot()
ax1 = plt.axes()
ax1.imshow(H.T, origin='lower', interpolation='nearest', aspect='auto',
          extent=[FeH_bins[0], FeH_bins[-1],
                  alphFe_bins[0], alphFe_bins[-1]],
          cmap=plt.cm.binary)
# plot cluster centers
cluster_centers = scaler.inverse_transform(clf.cluster_centers_)
ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           s=40, c='r', edgecolors='r')

# plot cluster boundaries
FeH_centers = 0.5 * (FeH_bins[1:] + FeH_bins[:-1])
alphFe_centers = 0.5 * (alphFe_bins[1:] + alphFe_bins[:-1])

Xgrid = np.meshgrid(FeH_centers, alphFe_centers)
Xgrid = np.array(Xgrid).reshape((2, 50 * 50)).T

H = clf.predict(scaler.transform(Xgrid)).reshape((50, 50))

for i in range(n_clusters):
    Hcp = H.copy()
    flag = (Hcp == i)
    Hcp[flag] = 1
    Hcp[~flag] = 0

#ax1 = fig.add_subplot(221)
#ax1.plot(galaxies['gRaw'] - galaxies['rRaw'],
#         galaxies['rRaw'],
#         **plot_kwargs)
#
#ax2 = fig.add_subplot(223, sharex=ax1)
#ax2.plot(galaxies['gRaw'] - galaxies['rRaw'],
#         galaxies['rRaw'] - galaxies['iRaw'],
#         **plot_kwargs)
#
#ax3 = fig.add_subplot(222, sharey=ax1)
#ax3.plot(stars['gRaw'] - stars['rRaw'],
#         stars['rRaw'],
#         **plot_kwargs)
#
#ax4 = fig.add_subplot(224, sharex=ax3, sharey=ax2)
#ax4.plot(stars['gRaw'] - stars['rRaw'],
#         stars['rRaw'] - stars['iRaw'],
#         **plot_kwargs)
#
## set labels and titles
#ax1.set_ylabel('$r$')
#ax2.set_ylabel('$r-i$')
#ax2.set_xlabel('$g-r$')
#ax4.set_xlabel('$g-r$')
#ax1.set_title('Galaxies')
#ax3.set_title('Stars')
#
### set axis limits
#ax2.set_xlim(-0.5, 3)
#ax3.set_ylim(22.5, 14)
#ax4.set_xlim(-0.5, 3)
#ax4.set_ylim(-1, 2)

## adjust tick spacings on all axes
#for ax in (ax1, ax2, ax3, ax4):
#for ax in (ax2):

ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
#ax1.xaxis.set_major_locator(plt.MultipleLocator(0.3))

plt.savefig('capstone_part2clustertest.png')
