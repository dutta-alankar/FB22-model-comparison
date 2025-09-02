#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:07:19 2025

@author: alankar
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pickle

dark = False
plt.cla()
plt.close()

## Plot Styling
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = False
matplotlib.rcParams["ytick.right"] = False
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.linestyle"] = ":"
matplotlib.rcParams["grid.linewidth"] = 0.8
matplotlib.rcParams["grid.color"] = "gray" if not(dark) else "white"
matplotlib.rcParams["grid.alpha"] = 0.3 if not(dark) else 0.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["legend.handletextpad"] = 0.4
matplotlib.rcParams["axes.linewidth"] = 1.0
matplotlib.rcParams["lines.linewidth"] = 3.5
matplotlib.rcParams["ytick.major.width"] = 1.2
matplotlib.rcParams["xtick.major.width"] = 1.2
matplotlib.rcParams["ytick.minor.width"] = 1.0
matplotlib.rcParams["xtick.minor.width"] = 1.0
matplotlib.rcParams["ytick.major.size"] = 11.0
matplotlib.rcParams["xtick.major.size"] = 11.0
matplotlib.rcParams["ytick.minor.size"] = 5.0
matplotlib.rcParams["xtick.minor.size"] = 5.0
matplotlib.rcParams["xtick.major.pad"] = 10.0
matplotlib.rcParams["xtick.minor.pad"] = 10.0
matplotlib.rcParams["ytick.major.pad"] = 6.0
matplotlib.rcParams["ytick.minor.pad"] = 6.0
matplotlib.rcParams["xtick.labelsize"] = 26.0
matplotlib.rcParams["ytick.labelsize"] = 26.0
matplotlib.rcParams["axes.titlesize"] = 28.0
matplotlib.rcParams["axes.labelsize"] = 35.0
matplotlib.rcParams["axes.labelpad"] = 8.0
plt.rcParams["font.size"] = 28
matplotlib.rcParams["legend.handlelength"] = 2
# matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["figure.figsize"] = (13,10)
if dark:
    plt.style.use('dark_background')

gamma = 5/3.

cool_table = np.loadtxt("cooltable.dat")
cc85_table = np.loadtxt("CC85_steady-prof_gamma_1.667.txt")

LAMBDA   = interp1d(cool_table[:,0], cool_table[:,1], fill_value="extrapolate")
cc85_rho = interp1d(cc85_table[:,0], cc85_table[:,1], fill_value="extrapolate")
cc85_prs = interp1d(cc85_table[:,0], cc85_table[:,2], fill_value="extrapolate")
cc85_vel = interp1d(cc85_table[:,0], cc85_table[:,3], fill_value="extrapolate")
cc85_mach = cc85_table[:,3]/np.sqrt(gamma*cc85_table[:,2]/cc85_table[:,1])
relpos = interp1d(cc85_mach, cc85_table[:,0]) #inverting the Mach relation

mp  = 1.6726e-24
pc  = 3.0856775807e18
kpc = 1e3*pc
kB  = 1.3806505e-16
yr  = 365*24*60**2
Myr = 1e6*yr
MSun = 2.0e+33

X_solar = 0.7154
Y_solar = 0.2703
Z_solar = 0.0143
fracZ   = 1.0
Xp      = X_solar*(1-fracZ*Z_solar)/(X_solar+Y_solar)
Yp = Y_solar*(1-fracZ*Z_solar)/(X_solar+Y_solar)
Zp = fracZ*Z_solar
mu     = 1./(2*Xp+0.75*Yp+0.5625*Zp);
mup    = 1./(2*Xp+0.75*Yp+(9./16.)*Zp);
muHp   = 1./Xp
mue    = 2./(1+Xp)
mui    = 1./(1/mu-1/mue)

# choose from this
data_sims = [(0.10, 35.335), (0.30, 106.006), (0.20, 70.671),
             (0.5, 176.677), (0.8, 282.684), 
             (1.00, 353.355), (1.40, 494.697),
             (2.50, 883.387), (8.00, 2292.516)]

chi_0 = 100
mach_0 = 1.496
tcoolmBytcc, dinibyRcl = data_sims[4] # vary this
# dinibyRcl = 282.684 #176.677 # vary this
dinibydinj = float(relpos(mach_0))
Tcl = 4.0e+04
PinibykB = 2.020e+06 # Kcm^-3, degenerate


Tw0 = chi_0*Tcl
Pw0 = PinibykB*kB
vw0 = mach_0*np.sqrt(gamma*kB*Tw0/(mu*mp))

rhoTini = cc85_rho(dinibydinj)
prsTini = cc85_prs(dinibydinj)
velTini = cc85_vel(dinibydinj)

alpha_go = 1.0
Rgo    = ( 10.378 * (Tcl/1e4)**(5/2)*mach_0/
          ((PinibykB/1e3)*(LAMBDA(np.sqrt(chi_0)*Tcl)/10**-21.29) ) * 
          (chi_0/100) * (alpha_go**-1) ) # pc
rcl_0  = (tcoolmBytcc**-1) * Rgo # pc
dini   = dinibyRcl*rcl_0 # pc
dinj   = dini/dinibydinj # pc
dinjbyRcl = dinj/rcl_0

Mdot = ((Pw0/prsTini)* (vw0/velTini)**(-1) *(dinj*pc)**2) / (MSun/yr)
Edot = ((Pw0/prsTini) * (vw0/velTini) *(dinj*pc)**2) #erg s^-1

UNIT_LENGTH   = rcl_0 * pc
UNIT_DENSITY  = rhoTini * ((Mdot*(MSun/yr))**1.5) * (Edot**-0.5) * ((dinj*pc)**-2)
UNIT_VELOCITY = velTini * ((Mdot*(MSun/yr))**-0.5) * (Edot**0.5)


tcool_0 = ( (gamma/(gamma-1))*cc85_prs(dinibydinj)*UNIT_DENSITY*UNIT_VELOCITY**2/ 
          ((chi_0**0.5 * cc85_rho(dinibydinj)*UNIT_DENSITY*Xp/mp)**2*
           LAMBDA(chi_0**0.5 * Tcl)) ) # cgs

rcl_0 = 1.0 # back to code units


def cloud_model(t, state, vanilla, _drummond = True):
    fcool = 2.0
    fmix  = 2.0
    fturb = 0.1
    Cdrag = 0.5
    test  = 0
    if _drummond:
        fcool = 2
        fmix  = 2
        fturb = 0.1
        Cdrag = 0.5
    
    dcl, vcl, Mcl = state
    # everything in code units
    
    dcl_dot = vcl
    
    rhow = cc85_rho(dcl/dinjbyRcl)/cc85_rho(dinibydinj) if not(vanilla) else 1.0
    prsw = cc85_prs(dcl/dinjbyRcl)/(cc85_rho(dinibydinj)*cc85_vel(dinibydinj)**2) if not(vanilla) else 1.0/(gamma*mach_0**2)
    rhocl = (prsw * UNIT_DENSITY*UNIT_VELOCITY**2 * (mu*mp)/(kB*Tcl))/UNIT_DENSITY
    chi = rhocl/rhow
    vw  = cc85_vel(dcl/dinjbyRcl)/cc85_vel(dinibydinj) if not(vanilla) else 1.0
    # chi_test = chi_0 * ((dcl/dinibyRcl)**(-2*(gamma-1)) if not(vanilla) else 1.0)
    # print(chi/chi_test
    rcl = (Mcl/(4*np.pi/3.*chi*rhow))**(1/3.) * ((dcl/dinibyRcl)**test if not(vanilla) else 1.0)
    # rcl_par = rcl_0 + min(t/(3*100*rcl_0/vw), 1) * (chi/2-1) * rcl_0
    # rcl_prp = np.sqrt(Mcl/(2*np.pi*chi*rhow*rcl_par))
    
    Tw = (prsw/rhow)*(mu*mp/kB)*UNIT_VELOCITY**2
    cs_hot = np.sqrt(gamma*kB*Tw/(mu*mp))/UNIT_VELOCITY
    cs_cl  = np.sqrt(gamma*kB*Tcl/(mu*mp))/UNIT_VELOCITY
    vrel = vw - vcl
    Tmix = np.sqrt(Tcl*Tw)
    rhomix = (prsw * UNIT_DENSITY*UNIT_VELOCITY**2 * (mu*mp)/(kB*Tmix))/UNIT_DENSITY
    tcool = ( (gamma/(gamma-1))*prsw*UNIT_DENSITY*UNIT_VELOCITY**2/ 
              ((rhomix*UNIT_DENSITY*Xp/mp)**2*
               LAMBDA(Tmix)) )/(UNIT_LENGTH/UNIT_VELOCITY)
    tcool_cl = ( (gamma/(gamma-1))*prsw*UNIT_DENSITY*UNIT_VELOCITY**2/ 
                 ((rhocl*UNIT_DENSITY*Xp/mp)**2*
                  LAMBDA(Tcl)) )/(UNIT_LENGTH/UNIT_VELOCITY)
    if tcool<0:
        tcool = 14*1e+03*Myr
    if tcool_cl<0:
        tcool_cl = 14*1e+03*Myr
    # print(rcl)
    # sat_vel = max(np.fabs(vrel), cs_cl if not _drummond else 0.)
    # if not(vanilla) and sat_vel<=1.4*cs_cl and not _drummond:
    #     fcool = chi**0.5
    
    # if vanilla and not _drummond and sat_vel<=1.0*cs_cl:
    #     fcool = min(fcool*chi**0.5*rcl**2, chi_0*rcl_0**2)/(chi**0.5*rcl**2)
    # (dcl/(dinibydinj/dinjbyRcl))
    if not _drummond and not vanilla:
        if dcl/(dinibydinj*dinjbyRcl)>=1:
            xi = rcl_0*(dcl/(dinibydinj*dinjbyRcl)) /(fturb*vrel*tcool)
        else:
            xi = rcl /(fturb*vrel*tcool)
    else:
        xi = rcl /(fturb*vrel*tcool)
    # if not(vanilla) and xi>12:
    #     print(dcl, xi)
    '''
    if vanilla:
        fdim = 0.64 if xi >= 4 else 0.3 #2/3. # dim = 2 + fdim
    else:
        fdim = 0.5 if xi >= 4 else 0.3 # dim = 2 + fdim
    '''
    if vanilla:
        fdim = 1/2. if xi >= 1. else 2/3. #2/3. # dim = 2 + fdim
    else:
        fdim = 1/2. if xi >= 1.0 else 2/3. # dim = 2 + fdim
    alpha = (3*fdim-1)/2.
    if _drummond:
        alpha = 0.25 if xi >= 1. else 0.50
    if not _drummond:
        if vanilla:
            vin = max(fturb * vrel * xi**alpha, 0.075*cs_cl*(rcl_0/(cs_cl*tcool_cl))**0.25)
        else:
            vin = max(fturb * vrel * xi**alpha, 0.19*cs_cl*(rcl/(cs_cl*tcool_cl))**0.25)
        # if vin>(fturb * vrel * xi**alpha):
        #     print(t/chi_0**0.5,"\n\n")
    else:
        vin = fturb * vrel * xi**alpha
    # Mcl_dot_0 = 3*fturb*fcool * (Mcl*np.fabs(vrel)/(chi**0.5*rcl)) 
    # if not(vanilla) and vin>(fturb * vrel * xi**alpha) and not _drummond:
    #     fcool = chi**0.5
    Mcl_dot_grow = 3*Mcl*fcool/(chi**0.5*rcl)*vin
    # Mcl_dot = Mcl_dot_0 * (xi**alpha*(sat_vel/np.fabs(vrel)) - fmix/fcool)
    Mcl_dot_loss = 3*fturb*fmix*(Mcl*vrel/(chi**0.5*rcl))
    Mcl_dot = Mcl_dot_grow - Mcl_dot_loss
    vcl_dot = vrel * (Mcl_dot_grow/Mcl) + (3/8.) * Cdrag * vrel**2 / (chi*rcl)
    # if t==0.0:
    #     print(dcl_dot, vcl_dot, Mcl_dot)
    return (dcl_dot, vcl_dot, Mcl_dot)

tcc_0 = np.sqrt(chi_0)
tstop = 149*tcc_0
time = np.arange(0., tstop, 1.0*tcc_0)

# our model
_drummond = False
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_cc85mod, vcl_cc85mod, Mcl_cc85mod = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod, vcl_vanmod, Mcl_vanmod = solution.y

# load sim data
vanilla = False
label = f"c{chi_0:d},m{mach_0:.3f},T4e4,t{tcoolmBytcc:.2f},r{dinibyRcl:.3f}"
if tcoolmBytcc==0.1:
    till = 25
elif tcoolmBytcc==0.2:
    till = 62
else:
    till = int(tstop/tcc_0)

directory = '/Users/alankard/Documents/Documents - MacBook-Pro-MPA/cc85-freya/python-scripts/analysis-scripts'
with open(f"{directory}/paraview-cloud-analysis_data-dump/{label}.pickle", "rb") as handle:
    cloud_data = pickle.load(handle)

cc85_from_anaysis = False

if not(cc85_from_anaysis):
    cc85_sim = []
    for key in list(cloud_data.keys()):
        cloud_density = cloud_data[key]['cloud_density'] * UNIT_DENSITY # cgs
        cloud_volume = cloud_data[key]['cloud_volume_elems'] * UNIT_LENGTH**3 # cgs
    
        cloud_mass = np.sum(cloud_density * cloud_volume)/MSun
        cloud_distance = np.sum(cloud_density * cloud_volume * 
                                cloud_data[key]['cloud_distance'])/(cloud_mass*MSun)
        cc85_sim.append([float(key), cloud_mass, 
                          cloud_distance,
                          cloud_data[key]['cloud_tot_surface_area']])
    cc85_sim = np.array(cc85_sim)
else:
  label_cc85 = f"../../output-{label}"
  data = np.loadtxt(f"{directory}/{label_cc85}/analysis.dat")
  cc85_sim = np.zeros((data.shape[0], 2))
  cc85_sim[:,0] = data[:,0]
  cc85_sim[:,1] = data[:,15]
  

vanilla = True
label_van = f"../../output-vanl-{label}"
van_mass_data = np.loadtxt(f"{directory}/{label_van}/analysis.dat")

cc85_color = "teal"
van_color = "goldenrod"

# -----------
# plot starts
# -----------
'''
# cloud area with distance
vanilla = False
rhow = cc85_rho(dcl_cc85mod/dinjbyRcl)/cc85_rho(dinibydinj)
chi = chi_0 * ((dcl_cc85mod/dinibyRcl)**(-2*(gamma-1)) if not(vanilla) else 1.0)
rcl = (Mcl_cc85mod/(4*np.pi/3.*chi*rhow))**(1/3.)

line, = plt.plot(dcl_cc85mod/dcl_cc85mod[0], (rcl/rcl[0])**2, 
                 linestyle="--" if vanilla else '-',
                 color = 'tab:cyan' if dark else 'tab:red',
                 label = f"{'vanilla' if vanilla else 'CC85'}")
plt.plot(dcl_cc85mod/dcl_cc85mod[0], 1*(dcl_cc85mod/dcl_cc85mod[0])**4, 
         color = 'tab:red' if dark else 'tab:red',
         linestyle = ':')

plt.plot(dcl_cc85mod/dcl_cc85mod[0], 4*(dcl_cc85mod/dcl_cc85mod[0])**4, 
         color = 'tab:red' if dark else 'tab:red',
         linestyle = ':')

vanilla = True
rhow = cc85_rho(dcl_vanmod/dinjbyRcl)/cc85_rho(dinibydinj)
chi = chi_0 * ((dcl_vanmod/dinibyRcl)**(-2*(gamma-1)) if not(vanilla) else 1.0)
rcl = (Mcl_vanmod/(4*np.pi/3.*chi*rhow))**(1/3.)

line, = plt.plot(dcl_vanmod/dcl_vanmod[0], (rcl/rcl[0])**2, 
                 linestyle="--" if vanilla else '-',
                 color = line.get_color(), 
                 label = f"{'vanilla' if vanilla else 'CC85'}")

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Cloud distance [$d_{\rm cl,ini}$]")
plt.ylabel(r"Cloud area $A_{\rm cold}$ [$A_{\rm cold, ini}$]")
plt.legend(loc="best", 
           # title=r"$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$", 
           ncols=3,
           prop = { "size": 24 }, 
           title_fontsize=26, 
           fancybox=True)

line, = plt.plot(cc85_sim[:,2]/cc85_sim[0,2], cc85_sim[:,3]/cc85_sim[0,3], 
                 color = 'white' if dark else 'black',
                 label=f"{'vanilla' if vanilla else 'CC85'} sim",
                 linestyle = '--' if vanilla else '-')
plt.savefig(f'distance-area_{"FB22" if _drummond else ""}.png', bbox_inches="tight")
plt.show()
'''

x1, x2, y1, y2 = 0, 100, 0.8, 25
axins = plt.gca().inset_axes([0.08, 0.67, 0.3, 0.3], xlim=(x1,x2), ylim=(y1,y2),
                             # xticklabels=[], 
                             # yticklabels=[]
                             )

# cloud mass with time

vanilla = False
line, = plt.plot(time/tcc_0, Mcl_cc85mod/Mcl_cc85mod[0],
                 linestyle="-",
                 color = cc85_color,
                 label = "CC85 model",
                 alpha = 1.0)
axins.plot(time/tcc_0, Mcl_cc85mod/Mcl_cc85mod[0],
           linestyle="-",
           color = cc85_color,
           alpha = 1.0)
# cc85_color = line.get_color()

vanilla = True
line, = plt.plot(time/tcc_0, Mcl_vanmod/Mcl_vanmod[0], 
                 linestyle="-",
                 color = van_color, 
                 label = "vanilla model",
                 alpha = 1.0)
'''
axins.plot(time/tcc_0, Mcl_vanmod/Mcl_vanmod[0], 
           linestyle="-",
            label = "vanilla model",
            color = van_color,
            alpha = 1.0)
'''
# van_color = line.get_color()

# normalize by cc85_sim[0,1]
vanilla = False
line, = plt.plot(cc85_sim[:till+1,0], cc85_sim[:till+1,1]/cc85_sim[0,1], 
                 color = cc85_color,
                 label="CC85 sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
axins.plot(cc85_sim[:till+1,0], cc85_sim[:till+1,1]/cc85_sim[0,1], 
                 color = cc85_color,
                 linestyle = (0, (3, 3)),
                 linewidth = 7)

# normalize by van_mass_data[0,1]
vanilla = True
line, = plt.plot(van_mass_data[:,0]/tcc_0, van_mass_data[:,5]/van_mass_data[0,5], 
                 color = van_color,
                 label="vanilla sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
'''
axins.plot(van_mass_data[:,0]/tcc_0, van_mass_data[:,5]/van_mass_data[0,5], 
           color = van_color,
           linestyle = (0, (3, 3)),
           linewidth = 7)
'''

# drummond model
_drummond = True
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_cc85mod_dr, vcl_cc85mod_dr, Mcl_cc85mod_dr = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod_dr, vcl_vanmod_dr, Mcl_vanmod_dr = solution.y

vanilla = False
line, = plt.plot(time/tcc_0, Mcl_cc85mod_dr/Mcl_cc85mod_dr[0],
                 linestyle="-",
                 color = cc85_color,
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4,
                 linewidth = 12)
# cc85_color = line.get_color()
axins.plot(time/tcc_0, Mcl_cc85mod_dr/Mcl_cc85mod_dr[0],
           linestyle="-",
           color = cc85_color,
           alpha = 0.4,
           linewidth = 12)

vanilla = True
line, = plt.plot(time/tcc_0, Mcl_vanmod_dr/Mcl_vanmod_dr[0], 
                 linestyle="-",
                 color = van_color, 
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4,
                 linewidth = 12)
# van_color = line.get_color()
'''
axins.plot(time/tcc_0, Mcl_vanmod_dr/Mcl_vanmod_dr[0], 
           linestyle="-",
           color = van_color, 
           alpha = 0.4,
           linewidth = 12)
'''

plt.text(52, 1.5, "Dark shade: our parameters\nLight shade: FB22 parameters")

# inset plot sub-region

high_res_cc85 = np.loadtxt("mass-time-t0.8-res16.txt")

axins.plot(high_res_cc85[:,0], high_res_cc85[:,1], 
           color = cc85_color,
           linestyle = ":",
           linewidth = 7)
axins.set_yscale("log")
# plt.gca().indicate_inset_zoom(axins, edgecolor="black")

def time_code2Myr(x):
    return x * tcc_0 * UNIT_LENGTH/UNIT_VELOCITY/Myr


def time_Myr2code(x):
    return x * Myr / (UNIT_LENGTH/UNIT_VELOCITY) / tcc_0


def mass_code2Msun(x):
    return x * (chi_0*4*np.pi) * UNIT_DENSITY*UNIT_LENGTH**3/MSun


def mass_Msun2code(x):
    return x * MSun / (UNIT_DENSITY*UNIT_LENGTH**3) / (chi_0*4*np.pi)

def length_dini2kpc(x):
    return x * dini/1.0e+03
    

def length_kpc2dini(x):
    return x * 1.0e+03/dini


secax = plt.gca().secondary_xaxis('top', functions=(time_code2Myr, time_Myr2code))
secax.set_xlabel(r"time [Myr]")

secay = plt.gca().secondary_yaxis('right', functions=(mass_code2Msun, mass_Msun2code))
secay.set_ylabel(r"Cloud mass $M_{\rm cold}$ [$M_{\odot}$]")

plt.yscale('log')
plt.xlim(xmin=0., xmax=tstop/tcc_0)
plt.ylim(ymin=0.4, ymax=205.0)
plt.title(r'$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$ = '+f'{tcoolmBytcc:.2f}', size=45, pad=20)
plt.xlabel(r"time [$t_{\rm cc,ini}$]")
plt.ylabel(r"Cloud mass $M_{\rm cold}$ [$M_{\rm cold, ini}$]")

plt.legend(loc="lower right", 
           # title=r"$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$", 
           ncols=2,
           prop = { "size": 24 }, 
           title_fontsize=26, 
           fancybox=True)
plt.savefig('mass-time_FB22vsOur-cloud.svg', bbox_inches="tight")
plt.show()
plt.close()
plt.cla()

'''
# additional plot
line, = plt.plot(cc85_sim[:,2]/cc85_sim[0,2], cc85_sim[:,3]/cc85_sim[0,3], 
                 color = 'white' if dark else 'black',
                 label=f"{'vanilla' if vanilla else 'CC85'} sim",
                 linestyle = '--' if vanilla else '-')
line, = plt.plot(cc85_sim[:,2]/cc85_sim[0,2], 4*(cc85_sim[:,2]/cc85_sim[0,2])**4 )
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
# time distance
# x1, x2, y1, y2 = 0, 50, 0.8, 11
# axins = plt.gca().inset_axes([0.08, 0.68, 0.3, 0.3], xlim=(x1,x2), ylim=(y1,y2),
#                              # xticklabels=[], 
#                              # yticklabels=[]
#                              )
# -------------------------------
# cloud distance with time
# -------------------------------
# our model
_drummond = False
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_cc85mod, vcl_cc85mod, Mcl_cc85mod = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod, vcl_vanmod, Mcl_vanmod = solution.y

vanilla = False
line, = plt.plot(time/tcc_0, dcl_cc85mod/dcl_cc85mod[0],
                 linestyle="-",
                 color = cc85_color,
                 label = "CC85 model",
                 alpha = 1.0)
# axins.plot(time/tcc_0, dcl_cc85mod/dcl_cc85mod[0],
#            color = cc85_color,
#            linestyle="-",
#            alpha = 1.0)

vanilla = True
line, = plt.plot(time/tcc_0, dcl_vanmod/dcl_vanmod[0], 
                 linestyle="-",
                 color = van_color, 
                 label = "vanilla model",
                 alpha = 1.0)
# axins.plot(time/tcc_0, Mcl_vanmod/Mcl_vanmod[0], 
#             linestyle="-",
#             color = van_color, 
#             label = "vanilla model",
#             alpha = 1.0)

# normalize by cc85_sim[0,1]
vanilla = False
line, = plt.plot(cc85_sim[:till+1,0], cc85_sim[:till+1,2]/cc85_sim[0,2], 
                 color = cc85_color,
                 label="CC85 sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
# axins.plot(cc85_sim[:till+1,0], cc85_sim[:till+1,2]/cc85_sim[0,2], 
#                  color = cc85_color,
#                  linestyle = (0, (3, 3)),
#                  linewidth = 7)

# normalize by van_mass_data[0,1]
vanilla = True
line, = plt.plot(van_mass_data[:,0]/tcc_0, van_mass_data[:,1]/van_mass_data[0,1], 
                 color = van_color,
                 label="vanilla sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
# axins.plot(van_mass_data[:,0]/tcc_0, van_mass_data[:,1]/van_mass_data[0,1], 
#            color = van_color,
#            linestyle = (0, (3, 3)),
#            linewidth = 7)

# drummond model
_drummond = True
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )

dcl_cc85mod_dr, vcl_cc85mod_dr, Mcl_cc85mod_dr = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod_dr, vcl_vanmod_dr, Mcl_vanmod_dr = solution.y

vanilla = False
line, = plt.plot(time/tcc_0, dcl_cc85mod_dr/dcl_cc85mod_dr[0],
                 linestyle="-",
                 color = cc85_color,
                 lw=12,
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4)
# cc85_color = line.get_color()
# axins.plot(time/tcc_0, dcl_cc85mod_dr/dcl_cc85mod_dr[0],
#            linestyle="-",
#            color = cc85_color,
#            alpha = 0.4)

vanilla = True
line, = plt.plot(time/tcc_0, dcl_vanmod_dr/dcl_vanmod_dr[0], 
                 linestyle="-",
                 color = van_color, 
                 lw=12, 
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4)

# van_color = line.get_color()
# axins.plot(time/tcc_0, dcl_vanmod_dr/dcl_vanmod_dr[0], 
#            linestyle="-",
#            color = van_color, 
#            alpha = 0.4)

plt.text(10, 7.0, "Dark shade: our parameters\nLight shade: FB22 parameters")

# inset plot sub-region

# high_res_cc85 = np.loadtxt("mass-time-t0.8-res16.txt")

# axins.plot(high_res_cc85[:,0], high_res_cc85[:,1], 
#            color = cc85_color,
#            linestyle = ":",
#            linewidth = 7)
# axins.set_yscale("log")
# plt.gca().indicate_inset_zoom(axins, edgecolor="black")


secax = plt.gca().secondary_xaxis('top', functions=(time_code2Myr, time_Myr2code))
secax.set_xlabel(r"time [Myr]")

secay = plt.gca().secondary_yaxis('right', functions=(length_dini2kpc, length_kpc2dini))
secay.set_ylabel(r"distance $d_{\rm cl}$ (center of mass) [kpc]")

# plt.yscale('log')
plt.xlim(xmin=0., xmax=tstop/tcc_0)
plt.ylim(ymin=0.1)
plt.title(r'$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$ = '+f'{tcoolmBytcc:.2f}', size=45, pad=20)
plt.xlabel(r"time [$t_{\rm cc,ini}$]")
plt.ylabel(r"distance $d_{\rm cl}$ (center of mass) [$d_{\rm cl, ini}$]")

plt.legend(loc="lower right", 
           # title=r"$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$", 
           ncols=2,
           prop = { "size": 24 }, 
           title_fontsize=26, 
           fancybox=True)
plt.savefig('distance-time_FB22vsOur-cloud.svg', bbox_inches="tight")
plt.show()
plt.close()
plt.cla()


# -------------------------------
# cloud mass with distance
# -------------------------------
# our model
_drummond = False
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_cc85mod, vcl_cc85mod, Mcl_cc85mod = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod, vcl_vanmod, Mcl_vanmod = solution.y

vanilla = False
line, = plt.plot(dcl_cc85mod/dcl_cc85mod[0], Mcl_cc85mod/Mcl_cc85mod[0],
                 linestyle="-",
                 color = cc85_color,
                 label = "CC85 model",
                 alpha = 1.0)
# axins.plot(time/tcc_0, dcl_cc85mod/dcl_cc85mod[0],
#            color = cc85_color,
#            linestyle="-",
#            alpha = 1.0)

vanilla = True
line, = plt.plot(dcl_vanmod/dcl_vanmod[0], Mcl_vanmod/Mcl_vanmod[0], 
                 linestyle="-",
                 color = van_color, 
                 label = "vanilla model",
                 alpha = 1.0)
# axins.plot(time/tcc_0, Mcl_vanmod/Mcl_vanmod[0], 
#             linestyle="-",
#             color = van_color, 
#             label = "vanilla model",
#             alpha = 1.0)

# normalize by cc85_sim[0,1]
vanilla = False
line, = plt.plot(cc85_sim[:till+1,2]/cc85_sim[0,2], cc85_sim[:till+1,1]/cc85_sim[0,1], 
                 color = cc85_color,
                 label="CC85 sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
# axins.plot(cc85_sim[:till+1,0], cc85_sim[:till+1,2]/cc85_sim[0,2], 
#                  color = cc85_color,
#                  linestyle = (0, (3, 3)),
#                  linewidth = 7)

# normalize by van_mass_data[0,1]
vanilla = True
line, = plt.plot(van_mass_data[:,1]/van_mass_data[0,1], van_mass_data[:,5]/van_mass_data[0,5], 
                 color = van_color,
                 label="vanilla sim",
                 linestyle = (0, (3, 3)),
                 linewidth = 7)
# axins.plot(van_mass_data[:,0]/tcc_0, van_mass_data[:,1]/van_mass_data[0,1], 
#            color = van_color,
#            linestyle = (0, (3, 3)),
#            linewidth = 7)

# drummond model
_drummond = True
vanilla = False 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )

dcl_cc85mod_dr, vcl_cc85mod_dr, Mcl_cc85mod_dr = solution.y

vanilla = True 
solution = solve_ivp(cloud_model, 
                     [0., tstop], 
                     [dinibyRcl, 0., (4*np.pi/3.)*chi_0], 
                     t_eval = time,
                     args = (vanilla, _drummond) )
dcl_vanmod_dr, vcl_vanmod_dr, Mcl_vanmod_dr = solution.y

vanilla = False
line, = plt.plot(dcl_cc85mod_dr/dcl_cc85mod_dr[0], Mcl_cc85mod_dr/Mcl_cc85mod_dr[0],
                 linestyle="-",
                 color = cc85_color,
                 lw=12,
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4)
# cc85_color = line.get_color()
# axins.plot(time/tcc_0, dcl_cc85mod_dr/dcl_cc85mod_dr[0],
#            linestyle="-",
#            color = cc85_color,
#            alpha = 0.4)

vanilla = True
line, = plt.plot(dcl_vanmod_dr/dcl_vanmod_dr[0], Mcl_vanmod_dr/Mcl_vanmod_dr[0], 
                 linestyle="-",
                 color = van_color, 
                 lw=12, 
                 # label = f"{'vanilla' if vanilla else 'CC85'} model",
                 alpha = 0.4)

# van_color = line.get_color()
# axins.plot(time/tcc_0, dcl_vanmod_dr/dcl_vanmod_dr[0], 
#            linestyle="-",
#            color = van_color, 
#            alpha = 0.4)

plt.text(3.5, 1.8, "Dark shade: our parameters\nLight shade: FB22 parameters")

# inset plot sub-region

# high_res_cc85 = np.loadtxt("mass-time-t0.8-res16.txt")

# axins.plot(high_res_cc85[:,0], high_res_cc85[:,1], 
#            color = cc85_color,
#            linestyle = ":",
#            linewidth = 7)
# axins.set_yscale("log")
# plt.gca().indicate_inset_zoom(axins, edgecolor="black")


secay = plt.gca().secondary_yaxis('right', functions=(mass_code2Msun, mass_Msun2code))
secay.set_ylabel(r"Cloud mass $M_{\rm cold}$ [$M_{\odot}$]")

secax = plt.gca().secondary_xaxis('top', functions=(length_dini2kpc, length_kpc2dini))
secax.set_xlabel(r"distance $d_{\rm cl}$ (center of mass) [kpc]")

plt.yscale('log')
# plt.xscale('log')
plt.ylim(ymin=0.4, ymax=205.0)
plt.xlim(xmin=1.0)
plt.title(r'$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$ = '+f'{tcoolmBytcc:.2f}', size=45, pad=20)
plt.ylabel(r"Cloud mass $M_{\rm cold}$ [$M_{\rm cold, ini}$]")
plt.xlabel(r"distance $d_{\rm cl}$ (center of mass) [$d_{\rm cl, ini}$]")

plt.legend(loc="lower right", 
           # title=r"$t_{\rm cool, mix}/t_{\rm cc}|_{\rm ini}$", 
           ncols=2,
           prop = { "size": 24 }, 
           title_fontsize=26, 
           fancybox=True)
plt.savefig('mass-distance_FB22vsOur-cloud.svg', bbox_inches="tight")
plt.show()
plt.close()
plt.cla()