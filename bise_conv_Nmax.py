### IMPORTS ### 
import sys

from Func_Floquet_classes import *
from matplotlib import cm
from IPython.display import Math
from sympy.interactive import printing

import seaborn as sns
import pandas as pd

## Personalized plots
from matplotlib.lines import Line2D
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r"\usepackage{bm,amsmath,amsfonts,amssymb,bbold}"
from Func_general import *
# plotParams('paper')

plt.rcParams.update({
    "pgf.texsystem": "lualatex",
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
         r"\usepackage{amssymb}",   
         r"\usepackage{unicode-math}",   # unicode math setup
         r"\setromanfont[Scale=1.04]{Libertinus Serif}",  # serif font via preamble
         r"\setsansfont[Scale=1]{Libertinus Sans}",
         r"\setmonofont[Scale=.89]{Liberation Mono}",
         r"\setmathfont{Libertinus Math}"
    ]),
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif"
})

def set_size(width, height=None, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if height==None:
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = height * inches_per_pt

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


textwidth = 304.44289 #pt
pagewidth = 457.8024 #pt
textheight = 689.0 #pt

palette = sns.color_palette("colorblind6")


## Units conversion
import scipy.constants as sconst

###############################
### Bi2Se3 Hamiltonian:

def calcBi2Se3(paramsBi2Se3):
    pZ = SimpleNamespace(**paramsBi2Se3)
    v0eff = pZ.v0 * pZ.a1
    C0eff, C2eff, R1eff = (pZ.C0 + pZ.a3 * pZ.M0), (pZ.C2 + pZ.a3 * pZ.M2), (0.5 * pZ.R1 * pZ.a1)
    C0eff = 0  # we are setting onsite constant term to 0
    pB = dict(**paramsBi2Se3, C0eff=C0eff, C2eff=C2eff, R1eff=R1eff, v0eff=v0eff)
    return pB

paramsBi2Se3 = calcBi2Se3({'C0': -0.0083, 'C2': 30.4, 'M0': -0.28, 'M2': 44.5, 'v0': 3.33,
                           'R1': 50.6, 'a1': 0.99, 'a3': -0.15})  # in units of eV and Angstrom

omega_eV = 0.160 #eV
paramsPulse = dict(adim=1 / omega_eV,  polar='linearx', theta=0)
pB = SimpleNamespace(**paramsBi2Se3, **paramsPulse)
tau = np.sqrt(2) * 1.5

hbar_eVs = sconst.physical_constants['reduced Planck constant in eV s'][0]
t_THZ = omega_eV/(2*np.pi*hbar_eVs)*1e-12

## Define the Floquet t-t' Hamiltonian

par_fix = {'C_0' : pB.C0eff, 'C_2': pB.C2eff, 'v_0': pB.v0eff, 'R_1': pB.R1eff,'W': 1/0.160, 'T_p' : 1.}
params_env = {'mu_sp' : 3*tau, 'tau_sp' : tau, 'A_x': 0.15, 'A_y':0, 'A_z': 0}
params_ham = {'k_x' : 0.1, 'k_y' : 0.0}

ham_symbols = sp.symbols("W C_0 C_2 v_0 R_1 mu_sp tau_sp", real=True)
ham_dict = {str(b): b for b in ham_symbols}
locals().update( ham_dict)

N = 50  # Important: large N!
Nmax = 20

hBi2Se3_pulse = Hamiltonian_ttp(
    h0_k = W * (C_0 * s0 + C_2 * (k_x ** 2 + k_y ** 2) * s0
                     + v_0 * (k_y * sx - k_x * sy) + R_1 * ((k_x+1j*k_y) ** 3 + (k_x-1j*k_y) ** 3) * sz),
    par_var_ham = params_ham,
    par_fix = par_fix,
    Vxt= sp.sin(2 * sp.pi / T_p * t_sp),
    Vyt= sp.sin(2 * sp.pi / T_p * t_sp + sp.pi / 2), 
    Axenv = A_x * sp.exp(-((t_sp-mu_sp)/(tau_sp))**2),
    Ayenv = A_y * sp.exp(-((t_sp-mu_sp)/(tau_sp))**2) , 
    ham_symbols=ham_symbols,
    N = N) #Importat: add N large!

## Quantities for adim 
A0 = 1./(pB.v0eff*par_fix['W'])
kprop = pB.v0eff*par_fix['W']

psi0band, T = 0, 1.
ks = np.linspace(-0.1, 0.1, 64) + 1e-6
ts = np.linspace(0, 2.0 * 3.0 * tau * T, 12000) + 1e-6

solverL1 = []
data     = []

print("Running cb")

for i,ki in enumerate(ks):
    print("k_x = ", ki)
    print("Progress: ", i/len(ks)*100, "%")
    params_env = {'mu_sp' : 3*tau, 'tau_sp' : tau , 'A_x': A0*4, 'A_y':0, 'A_z': 0}
    params_ham = {'k_x' : ki, 'k_y' : 0.0,}
    solver_ki = IFS_solver(hBi2Se3_pulse, ts, params_env, params_ham, Nmax = Nmax)
    UsC= hBi2Se3_pulse.time_evolutionU(dict(**params_env, **params_ham, **par_fix), ts, steps = True)
    ct, psitsol = solver_ki.c_t(psi0band=psi0band, psi_t=True)
    ifs_hamiltonian = solver_ki.ChamL
    ts_ct = ts[:len(ct)]
    
    data.append({
        'k_x': ki * kprop,
        'tagvec':solver_ki.tag_fqlevels(), 
        'indexCbase' : solver_ki.indexCbase, 
        'hamiltonian':ifs_hamiltonian,
        'eL':solver_ki.eL, 
        'ct':ct, 
        'psit':psitsol, 
        'psiTDSE': UsC@psitsol[0], 
        'ts' : ts, 
        'ts_ct': ts_ct, **params_env, **params_ham})
    
path = '/home/how09898/phd/thesis-figures-data/floquet-sidebands/data-bise/'
filename = f'fig_Bi2Se3Lin_calphaData_conv_N_{N}_Nmax_{Nmax}_sin_cb.npy'
np.save(path+filename, data)

print("Data saved to ", path+filename)
