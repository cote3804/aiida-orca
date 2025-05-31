# pull a raman spectrum calc and plot it
from aiida.orm import load_node
import matplotlib.pyplot as plt
from aiida import load_profile
import numpy as np
import os

def plot_lines(color, norm_factor=1):
    ax = plt.gca()
    for raman, freq in zip(ramans, freqs):
        if freq > 1700:
            continue
        ax.vlines(freq, 0, raman*norm_factor, colors=color)

def gaussian(x, mean, height, sigma=1):
    """ Return the normalized Gaussian with standard deviation sigma. """
    return height * np.exp(-0.5 * ((x - mean)**2)/(2*sigma))

def plot_spectrum(color, label, norm_factor=1):
    ax = plt.gca()
    xvals = np.linspace(min(freqs), 1600, 1000)
    yvals = np.zeros_like(xvals)
    for raman, freq in zip(ramans, freqs):
        if freq > 1700:
            continue
        yvals += gaussian(xvals, freq, raman*norm_factor, sigma=50)
    ax.plot(xvals, yvals, color=color, label=label)

load_profile()
ramanworkchain = load_node(2526)
ramancalcs = [load_node(2534), load_node(2586)] # diglyme, Na + diglyme
ramancalcs = [load_node(2657), load_node(2736)] # hug mode diglyme, hug mode Na + diglyme
ramancalcs = [load_node(3264), load_node(3249)] # hug mode double diglyme
ramancalcs = [load_node(3249), load_node(2736), load_node(2657)] # hug mode double diglyme, hug mode single diglyme
ramancalcs = [load_node(2657), load_node(2736), load_node(3854)] # DG, DG + Na, NaPF6 +DG
ramancalcs = [load_node(2657), load_node(3249)] # DG, 2DG
ramancalcs = [load_node(4093), ] # NaBPH

fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, sharey=True)
colors = ['r', 'b']
colors = ['r', 'b', 'g']
labels = ['DG', 'DG+Na']
labels = ['2DG+Na', '1DG+Na', '1DG']
labels = ["DG", "Na + DG", "NaPF6 + DG"]
labels = ["NaBPh", "2DG"]

# double_glyme_vib_1 = ramancalcs[0].base.links.get_outgoing().get_node_by_label("output_parameters").get_dict()["vibramans"][59]
# double_glyme_vib_2 = ramancalcs[0].base.links.get_outgoing().get_node_by_label("output_parameters").get_dict()["vibramans"][60]
# single_glyme_vib = ramancalcs[1].base.links.get_outgoing().get_node_by_label("output_parameters").get_dict()["vibramans"][28]
# calculating an intensity normalization factor by dividing the single glyme mode intensity at 1158 cm^-1
# by the sum of the two intensities at 1167 cm^-1.
# print(single_glyme_vib)
# print(double_glyme_vib_1)
# intensity_ratio = single_glyme_vib / (double_glyme_vib_1 + double_glyme_vib_2)
# print(intensity_ratio)

for ir, ramancalc in enumerate(ramancalcs):
    results = ramancalc.base.links.get_outgoing().get_node_by_label("output_parameters")
    res_dict = results.get_dict()
    ramans = res_dict["vibramans"] # raman intensity. derived from change in polarizability w/ vibration
    freqs = res_dict["vibfreqs"]
    # for i, freq in enumerate(freqs):
    #     print(i, freq)
    #     print(i, ramans[i])
    color = colors[ir]
    label = labels[ir]
    norm_factor = 1
    # if ir in [0]:
    #     norm_factor = 1.8
    # else:
    #     norm_factor = 1
    plt.sca(ax[0])
    plot_spectrum(color, label, norm_factor=norm_factor)
    plot_lines(color, norm_factor=norm_factor)
    plt.sca(ax[1])
    plot_spectrum(color, label, norm_factor=norm_factor)
    plot_lines(color, norm_factor=norm_factor)

ax[0].set_xlim([1140, 1220])
ax[1].set_xlim([1400, 1550])

ax[0].spines.right.set_visible(False)
ax[1].spines.left.set_visible(False)
# ax[0].yaxis.tick_right()
ax[0].tick_params(labelright=False)
ax[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
# ax[1].set_yticks([])
ax[1].tick_params(axis="y", colors="w")
ax[0].set_ylabel("Intensity", fontsize=14)
ax[1].legend()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[0].plot([1, 1], [0, 1], transform=ax[0].transAxes, **kwargs)
ax[1].plot([0, 0], [1, 0], transform=ax[1].transAxes, **kwargs)

print(os.getcwd())
plt.savefig("NaBPh_spectra", dpi=400)
