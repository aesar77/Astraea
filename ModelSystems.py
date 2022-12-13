#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:05:01 2022

@author: lunaeaqua
"""
#Creates model systems from confirmed exoplanet systems. 
#Returns an object array of class exoplanet_system.

import numpy as np
import astropy.io.ascii as asc
import astropy.constants as cs
import astropy.units as u
import matplotlib.pyplot as plt
import scipy as sy

data = asc.read("/home/lunaeaqua/Downloads/exoplanet.eu_catalog.csv",format="csv", fast_reader=False, guess=False)
planets = data["# name"]
stars = data["star_name"]
periods = data["orbital_period"]
pl_radii = data["radius"]
st_radii = data["star_radius"]
semi_axs = data["semi_major_axis"]

class exoplanet_system():
    def __init__(self, planet,pl_rad, star, st_rad, semi_ax, period):
        self.planet = planet 
        self.Rp = pl_rad * 69911000 * u.m #converting from R_jupyter to meter
        self.star = star 
        self.Rs = st_rad * cs.R_sun #converting from R_sun to meter
        self.a = semi_ax * cs.au #converting from au to meter
        self.P = period  #converting from days to second
        self.T = 0
        
    def set_T(self, tr):
        self.T = tr
        


def import_system(reqmodel):
    modelnum = reqmodel * 3
    models = []
    for i in range(0,modelnum):
        #j = np.random.randint(0, len(data["# name"]))
        j = i
        models.append(exoplanet_system(planets[j], pl_radii[j], stars[j], st_radii[j], semi_axs[j], periods[j]))
    
    for m in models:
        T = (m.Rs)*(m.P) / (np.pi * m.a)
        m.set_T(T)
        
    selected_models = np.random.choice([m for m in models if m.T >0 and m.Rp > 0], reqmodel)
    print("model size: ", np.size(selected_models))
    return selected_models


def power_law(x):
    return sy.stats.beta.fit(x)

    
s = import_system(1000)



rratcomp_arr = []
rsmacomp_arr = []
pericomp_arr = []

for sys in s:
    rratcomp_arr.append(float(sys.Rp/sys.Rs))
    rsmacomp_arr.append(float((sys.Rp + sys.Rs)/sys.a ))
    pericomp_arr.append(float(sys.P))
    
pars1 = power_law(rratcomp_arr)
fit1 = sy.stats.beta.rvs(*pars1, size=len(rratcomp_arr))
pars2 = power_law(rsmacomp_arr)
fit2 = sy.stats.beta.rvs(*pars2, size=len(rsmacomp_arr))
pars3 = power_law(pericomp_arr)
fit3 = sy.stats.beta.rvs(*pars3, size=len(pericomp_arr))

plt.plot(rratcomp_arr)
plt.plot(fit1)
plt.show()

plt.rcParams["figure.figsize"] = (10,10)
plt.hist([rratcomp_arr,fit1])
plt.legend(["Model Data","Fit" ])
plt.title("Rratcomp Model and Estimates")
plt.xlabel("rratcomp")
plt.ylabel("Number of systems")
plt.show()

plt.hist([rsmacomp_arr,fit2])
plt.legend(["Model Data","Fit" ])
plt.title("Rsmacomp Model and Estimates")
plt.xlabel("rsmacomp")
plt.ylabel("Number of systems")
plt.show()

plt.hist([pericomp_arr,fit3])
plt.legend(["Model Data","Fit" ])
plt.title("Pericomp Model and Estimates")
plt.xlabel("pericomp")
plt.ylabel("Number of systems")
plt.show()

pars = [pars1, pars2, pars3]
