import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astrotools import container as ctn
from astrotools import statistics as stats
from astrotools import auger
from scipy.interpolate import interp1d

with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fancybox": False}
mpl.rcParams.update(with_latex)

# spectral index for different energy in log scale from 18.3 to 19.9
sp_index = [3.4, 3.3, 3.2, 3.15, 3.05, 2.8, 2.6, 2.5, 2.55, 2.5, 2.6, 2.55,
            2.75, 2.8, 2.95, 2.9, 3.1, 3.1, 3.15, 3., 3.35, 3.75, 3.95,
            4.25, 4.7, 4.3, 4.5,4.75]
sp_index = np.array(sp_index)

sp_bins = np.linspace(np.log10(2.5*pow(10,18)), 20.2, len(sp_index))
int_sp = interp1d(sp_bins, sp_index, kind='cubic')

evt_auger = [83143, 47500, 28657, 17843, 12435, 8715, 6050, 4111, 2620, 1691,
           991, 624, 372, 156, 83, 24, 9, 6]

evt_bins = np.linspace(np.log10(2.5*pow(10,18)), np.log10(.3*pow(10,20)), len(evt_auger))

#e =  np.linspace(np.log10(2.5*pow(10,18)), np.log10(.3*pow(10,20)), len(sp_index))
e_bins = np.arange(np.log10(2.5*pow(10,18)), 20.2, 0.1)

nuclei = "proton"
path_scratch = '/net/scratch/Adrianna/data_analysis/arrays/'
path_scratch_CIC = '/net/scratch/Adrianna/data_analysis_data/SIMULATIONS/arrays/'

#==== Naming npz arrays ====#
SD_array  = path_scratch + 'SD_only/%s_SDonly_merge_p3.npz' % nuclei
array  = path_scratch_CIC + "Ecalibrated_%s" % nuclei
all_mass = path_scratch_CIC + "E_all"

#==== Opening data containers ====#
SD_data  = ctn.DataContainer(SD_array)
data     = ctn.DataContainer(array)
all_data = ctn.DataContainer(all_mass)

Erec = all_data["ESD_rec"] #SD_data["SD_energy"]
EMC  = all_data["ESD_MC"] #SD_data["MC_energy"]

log10e_bins = np.arange(18., 20.25, 0.05)
mask = np.log10(EMC)>=18.
EMC = EMC[mask]
Erec = Erec[mask]

H = np.histogram(np.log10(EMC), bins=log10e_bins)

w = 1 / H[0][np.digitize(np.log10(EMC), bins=log10e_bins) - 1] # Correction for not having a flat spectrum

n_arrays = 5000 # Number of arrays (npy files)  to create

Erec_new = np.zeros((n_arrays, len(log10e_bins)-1))
EMC_new  = np.zeros((n_arrays, len(log10e_bins)-1))

rdm_indx = -1.5 -2*np.random.random(n_arrays) # randomizing the spectral index

for i in range(n_arrays):
    if i % 100 == 0:
        print(i)
    p = (EMC) **(rdm_indx[i] + 1) * w  # Differential spectrum
    rdm = np.random.choice(np.arange(len(Erec)), sum(evt_auger), replace=True, p=p/sum(p))

    Erec_new[i] = np.histogram(np.log10(Erec[rdm]), bins = log10e_bins)[0]
    EMC_new[i]  = np.histogram(np.log10(EMC[rdm]) , bins = log10e_bins)[0]

np.save("spectrum_label.npy", Erec_new)
np.save("spectrum_target.npy", EMC_new)
