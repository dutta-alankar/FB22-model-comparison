#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:02:45 2025

@author: alankar
"""

################################################################
# Author: Drummond Fielding
# Reference: Fielding & Bryan (2021)
# Date: 08 Aug 2021
# Brief: This code calculates the structure of multiphase galactic winds.
#
# Execution:
# >> python MultiphaseGalacticWind.py
#
# Output: a 9 panel figure showing the properties of a multiphase galactic wind relative to a single phase galactic wind 
# 
# Overview:
# - First the code calculates the structure of a single phase galactic wind in the manner of Chevalier and Clegg (1985). 
# - Then the code calculates the structure of a multiphase galactic wind. 
# - The default values are:
#   - SFR            = 20 Msun/yr   (star formation rate)
#   - eta_E          = 1            (energy loading)
#   - eta_M          = 0.1          (initial hot phase or single pahse mass loading)
#   - eta_M_cold     = 0.2          (initial cold phase mass loading)
#   - M_cloud_init   = 10^3 Msun    (initial cloud mass)
#   - v_cloud_init   = 10^1.5 km/s  (initial cloud velocity)
#   - r_sonic        = 300 pc       (sonic radius)
#   - Z_wind_init    = 2 * Z_solar  (initial wind metallicity)
#   - Z_cloud_init   = Z_solar      (initial cloud metallicity)
#   - v_circ0        = 150 km/s     (circular velocity of external isothermal gravitational potential)
#
################################################################

import numpy as np
import glob
import h5py 
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
import cmasher as cmr
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import pickle

## Plot Styling
dark = False

## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = False
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.linestyle"] = ":"
matplotlib.rcParams["grid.linewidth"] = 2
matplotlib.rcParams["grid.color"] = "gray" if not(dark) else "white"
matplotlib.rcParams["grid.alpha"] = 0.3 if not(dark) else 0.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 1.0
matplotlib.rcParams["axes.linewidth"] = 3.0
matplotlib.rcParams["lines.linewidth"] = 7.0
matplotlib.rcParams["ytick.major.width"] = 3.2
matplotlib.rcParams["xtick.major.width"] = 3.2
matplotlib.rcParams["ytick.minor.width"] = 2.0
matplotlib.rcParams["xtick.minor.width"] = 2.0
matplotlib.rcParams["ytick.major.size"] = 18.0
matplotlib.rcParams["xtick.major.size"] = 18.0
matplotlib.rcParams["ytick.minor.size"] = 9.0
matplotlib.rcParams["xtick.minor.size"] = 9.0
matplotlib.rcParams["xtick.major.pad"] = 15.0
matplotlib.rcParams["xtick.minor.pad"] = 15.0
matplotlib.rcParams["ytick.major.pad"] = 9.0
matplotlib.rcParams["ytick.minor.pad"] = 9.0
matplotlib.rcParams["xtick.labelsize"] = 42.0
matplotlib.rcParams["ytick.labelsize"] = 42.0
matplotlib.rcParams["axes.titlesize"] = 50.0
matplotlib.rcParams["axes.labelsize"] = 50.0
matplotlib.rcParams["axes.labelpad"] = 12.0
plt.rcParams["font.size"] = 38
matplotlib.rcParams["legend.handlelength"] = 8
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["figure.figsize"] = (40,32)
if dark:
    plt.style.use('dark_background')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')


## Defining useful constants
gamma   = 5/3.
kb      = 1.3806488e-16
mp      = 1.67373522381e-24
km      = 1e5
s       = 1
yr      = 3.1536e7
Myr     = 3.1536e13
Gyr     = 3.1536e16
pc      = 3.086e18
kpc     = 1.0e3 * pc
Msun    = 2.e33
mu      = 0.62 
muH     = 1/0.75
Z_solar = 0.02

log_M_cloud_init = 2.0 # float(sys.argv[1])
high_load = "" # "_high_load"


fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)


def plot_data(drummond):
    data = {}
    print(f'FB22_{"drumm" if drummond else "new-params"}_logMcold-init={log_M_cloud_init:.1f}.pickle')
    with open(f'FB22_{"drumm" if drummond else "new-params"}_logMcold-init={log_M_cloud_init:.1f}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    single_phase_color  = 'tab:gray'
    cloud_colors        = 'goldenrod' if drummond else 'teal'
    cs_linestyle        = ':'
    cloud_linestyle     = '--'
    
    if drummond:
        (r_hot_only_b_kpc, v_wind_hot_only_b_1e5, cs_wind_hot_only_b_1e5, 
         rho_wind_hot_only_b_mu_mp, P_wind_hot_only_b_kb, 
         Z_wind_init_b_Z_solar, K_wind_hot_only,
         Mdot_wind_hot_only,
         Edot_wind_hot_only ) = data["single_phase"].T
        
        axv.loglog(r_hot_only_b_kpc, v_wind_hot_only_b_1e5,      color = single_phase_color )
        axv.loglog(r_hot_only_b_kpc, cs_wind_hot_only_b_1e5,     color = single_phase_color , ls = cs_linestyle)
        axd.loglog(r_hot_only_b_kpc, rho_wind_hot_only_b_mu_mp,  color = single_phase_color )
        axP.loglog(r_hot_only_b_kpc, P_wind_hot_only_b_kb,       color = single_phase_color )
        axZ.semilogx(r_hot_only_b_kpc, Z_wind_init_b_Z_solar,    color = single_phase_color )
        axK.loglog(r_hot_only_b_kpc, K_wind_hot_only,            color = single_phase_color )
        axMd.loglog(r_hot_only_b_kpc, Mdot_wind_hot_only, color = single_phase_color )
        axEd.loglog(r_hot_only_b_kpc, Edot_wind_hot_only, color = single_phase_color )
    
    
    (r_b_kpc, v_wind_b_1e5, cs_wind_b_1e5, 
     v_cloud_b_1e5,
     rho_wind_b_mu_mp, P_wind_b_kb, rhoZ_wind_b_rho_wind_b_Z_solar,
     Z_cloud_b_Z_solar, 
     K_wind, 
     M_cloud_b_Msun, 
     cloud_ksi, 
     Mdot_wind, cloud_Mdots, 
     Edot_wind, 
     cloud_Edots ) = data["multi_phase"].T
    
    print(f"ksi_init ({'drumm' if drummond else 'new-params'}) = {cloud_ksi[0]:.2f}")
    
    axv.loglog(r_b_kpc, v_wind_b_1e5,                     color = cloud_colors )
    axv.loglog(r_b_kpc, cs_wind_b_1e5,                    color = cloud_colors ,ls = cs_linestyle )
    axv.loglog(r_b_kpc, v_cloud_b_1e5,                    color = cloud_colors ,ls = cloud_linestyle )
    axd.loglog(r_b_kpc, rho_wind_b_mu_mp,                 color = cloud_colors )
    axP.loglog(r_b_kpc, P_wind_b_kb,                      color = cloud_colors )
    axZ.semilogx(r_b_kpc, rhoZ_wind_b_rho_wind_b_Z_solar, color = cloud_colors )
    axZ.semilogx(r_b_kpc, Z_cloud_b_Z_solar,              color = cloud_colors ,ls = cloud_linestyle )
    axK.loglog(r_b_kpc, K_wind,                           color = cloud_colors )
    axM.loglog(r_b_kpc, M_cloud_b_Msun,                   color = cloud_colors )
    # if sol.status == 1:
    #     axM.scatter(r[-1]_b_kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]_b_Msun,  color = cloud_colors , marker='x', s=8)
    axX.loglog(r_b_kpc, cloud_ksi,     color = cloud_colors )
    axMd.loglog(r_b_kpc, Mdot_wind,    color = cloud_colors )
    axMd.loglog(r_b_kpc, cloud_Mdots,  color = cloud_colors , ls = cloud_linestyle)
    axEd.loglog(r_b_kpc, Edot_wind,    color = cloud_colors )
    axEd.loglog(r_b_kpc, cloud_Edots,  color = cloud_colors , ls = cloud_linestyle)
    
    if drummond:
        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]

        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cs_linestyle),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='best',
                   fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)

        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',
                   fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)


        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],
                    loc='best',fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)
        
        custom_lines = [Line2D([0], [0], color=single_phase_color, ls = '-'),
                        (Line2D([0], [0], color='goldenrod', ls = '-'), Line2D([0], [0], color='teal', ls = '-')) ] 
        axd.legend(custom_lines, [r'Adiabatic (CC85) wind', r'Multiphase wind models'],
                   handler_map={tuple: HandlerTuple(ndivide=None)},
                   loc='best',fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)


plot_data(drummond = True)
plot_data(drummond = False)

custom_lines = [Line2D([0], [0], color='goldenrod', ls = '-'),
                Line2D([0], [0], color='teal', ls = '-')]
axX.legend(custom_lines, [r'FB22 single-cloud prescription', r'prescription from this work'],loc='lower left',
           fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P/k_B \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=3, ls=':')

axK.set_ylim(ymin=4e6, ymax=2e8)
# axK.set_ylim(ymin=7e6, ymax=7e8)
# axK.set_ylim(ymin=8e6, ymax=2e8)
axM.set_ylim(ymin=10.**(log_M_cloud_init-0.8), ymax=10.**(log_M_cloud_init+0.8))
# axMd.set_ylim(ymin=0.08, ymax=10.**0.1)
# axMd.set_ylim(ymin=0.3, ymax=4.0)
axMd.set_ylim(ymin=0.03, ymax=7.0)
# axEd.set_ylim(ymin=4e4, ymax=3.5e6)
axEd.set_ylim(ymin=3e4, ymax=8e6)

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')
plt.savefig(f'FB22wOur-wind{high_load}.svg', bbox_inches='tight')
plt.show()
plt.clf()
matplotlib.rcParams["figure.figsize"] = (40,13)
fig, (axd,axP,axv) = plt.subplots(1,3, constrained_layout=True)

def plot_data_short(drummond, normalize=False):
    data = {}
    with open(f'FB22_{"drumm" if drummond else "new-params"}_logMcold-init={log_M_cloud_init:.1f}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    single_phase_color  = 'tab:gray'
    cloud_colors        = 'goldenrod' if drummond else 'teal'
    cs_linestyle        = ':'
    cloud_linestyle     = '--'
    
    if drummond or normalize:
        (r_hot_only_b_kpc, v_wind_hot_only_b_1e5, cs_wind_hot_only_b_1e5, 
         rho_wind_hot_only_b_mu_mp, P_wind_hot_only_b_kb, 
         Z_wind_init_b_Z_solar, K_wind_hot_only,
         Mdot_wind_hot_only,
         Edot_wind_hot_only ) = data["single_phase"].T
        
        if not(normalize):
            axv.loglog(r_hot_only_b_kpc, v_wind_hot_only_b_1e5,      color = single_phase_color )
            axv.loglog(r_hot_only_b_kpc, cs_wind_hot_only_b_1e5,     color = single_phase_color , ls = cs_linestyle)
            axd.loglog(r_hot_only_b_kpc, rho_wind_hot_only_b_mu_mp,  color = single_phase_color )
            axP.loglog(r_hot_only_b_kpc, P_wind_hot_only_b_kb,       color = single_phase_color )
    
    (r_b_kpc, v_wind_b_1e5, cs_wind_b_1e5, 
     v_cloud_b_1e5,
     rho_wind_b_mu_mp, P_wind_b_kb, rhoZ_wind_b_rho_wind_b_Z_solar,
     Z_cloud_b_Z_solar, 
     K_wind, 
     M_cloud_b_Msun, 
     cloud_ksi, 
     Mdot_wind, cloud_Mdots, 
     Edot_wind, 
     cloud_Edots ) = data["multi_phase"].T
    
    print(f"ksi_init ({'drumm' if drummond else 'new-params'}) = {cloud_ksi[0]:.2f}")
    
    if not(normalize):
        axv.loglog(r_b_kpc, v_wind_b_1e5,                     color = cloud_colors )
        axv.loglog(r_b_kpc, cs_wind_b_1e5,                    color = cloud_colors ,ls = cs_linestyle )
        axv.loglog(r_b_kpc, v_cloud_b_1e5,                    color = cloud_colors ,ls = cloud_linestyle )
        axd.loglog(r_b_kpc, rho_wind_b_mu_mp,                 color = cloud_colors )
        axP.loglog(r_b_kpc, P_wind_b_kb,                      color = cloud_colors )
    else:
        clip = r_b_kpc.shape[0]
        axv.loglog(r_b_kpc, v_wind_b_1e5/v_wind_hot_only_b_1e5[:clip],                     color = cloud_colors )
        axv.loglog(r_b_kpc, cs_wind_b_1e5/cs_wind_hot_only_b_1e5[:clip],                    color = cloud_colors ,ls = cs_linestyle )
        axv.loglog(r_b_kpc, v_cloud_b_1e5/v_wind_hot_only_b_1e5[:clip],                    color = cloud_colors ,ls = cloud_linestyle )
        axd.loglog(r_b_kpc, rho_wind_b_mu_mp/rho_wind_hot_only_b_mu_mp[:clip],                 color = cloud_colors )
        axP.loglog(r_b_kpc, P_wind_b_kb/P_wind_hot_only_b_kb[:clip],                      color = cloud_colors )
    
    if drummond:
        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]

        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cs_linestyle),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='best',
                   fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)

        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        
        custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                        Line2D([0], [0], color='k', ls = cloud_linestyle)]
        
        if not(normalize):
            custom_lines = [Line2D([0], [0], color=single_phase_color, ls = '-'),
                            (Line2D([0], [0], color='goldenrod', ls = '-'), 
                             Line2D([0], [0], color='teal', ls = '-')) ] 
            axd.legend(custom_lines, [r'Adiabatic (CC85) wind', 
                                      r'Multiphase wind models'],
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       loc='best',fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)
         
normalize = False
plot_data_short(drummond = True, normalize=normalize)
plot_data_short(drummond = False, normalize=normalize)

custom_lines = [Line2D([0], [0], color='goldenrod', ls = '-'),
                Line2D([0], [0], color='teal', ls = '-')]
axP.legend(custom_lines, [r'FB22 parameters', r'Our parameters'],loc='upper right',
           fontsize=45, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P/k_B \; [{\rm K cm}^{-3}]$')

if not(normalize):
    axv.set_xlim(xmin=0.1, xmax=11)
    axd.set_xlim(xmin=0.1, xmax=11)
    axP.set_xlim(xmin=0.1, xmax=11)
    
    axd.set_ylim(ymin=2e-5)
    axP.set_ylim(ymin=2e-1)

axv.set_xlabel(r'$r\; [{\rm kpc}]$')
axd.set_xlabel(r'$r\; [{\rm kpc}]$')
axP.set_xlabel(r'$r\; [{\rm kpc}]$')
plt.savefig(f'FB22wOur-wind{high_load}-short.svg', bbox_inches='tight')
plt.show()
plt.clf()