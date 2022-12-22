#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:33:26 2022

@author: lunaeaqua
"""

import numpy as np
import astropy.io.ascii as asc
import astropy.constants as cs
import astropy.units as u
import matplotlib.pyplot as plt
import scipy as sy

data = asc.read("/home/lunaeaqua/Downloads/nasaexoplanettransit.csv",format="csv", fast_reader=False, guess=False)
planets = data["pl_name"]
stars = data["hostname"]
periods = data["pl_orbper"]
pl_radii = data["pl_radj"]
st_radii = data["st_rad"]
semi_axs = data["pl_orbsmax"]

class exoplanet_system():
    def __init__(self, planet,pl_rad, star, st_rad, semi_ax, period):
        self.planet = planet 
        self.Rp = pl_rad * 69911000 * u.m #converting from R_jupyter to meter
        self.star = star 
        self.Rs = st_rad * cs.R_sun #converting from R_sun to meter
        self.a = semi_ax * cs.au #converting from au to meter
        self.P = period  #converting from days to second
        self.T = 0
        self.MajorRj = False
        
    def set_T(self, tr):
        self.T = tr
        
    def set_Rj(self, Rj):
        self.MajorRj = Rj
        


def import_system(reqmodel, rtype):
    modelnum = reqmodel
    models = []

    for i in range(0,modelnum):
        #j = np.random.randint(0, len(data["# name"]))
        j = i
        models.append(exoplanet_system(planets[j], pl_radii[j], stars[j], st_radii[j], semi_axs[j], periods[j]))
    
    for m in models:
        T = (m.Rs)*(m.P) / (np.pi * m.a)
        m.set_T(T)
        if m.Rp >  69911000 * u.m:
            m.set_Rj(True)
     
    if rtype == "allR":
        
        selected_models = np.random.choice([m for m in models if m.T >0 and m.Rp > 0], reqmodel)
        print("model size: ", np.size(selected_models), " model type: ", rtype[1])
        return selected_models
    else:
        selected_models = [m for m in models if m.T >0 and m.MajorRj]
        print("model size: ", np.size(selected_models), " model type: ", rtype[1])
        return selected_models       


def power_law(x):
    return sy.stats.beta.fit(x)

 
allR = "allR", "All Radii"
majRj = "majRj", "Rp $>$ Rj"
model_type = [allR, majRj]
rtype = model_type[1]

#3938 at total, only transiting ones are choosen
size = 3938  
s = import_system(size, rtype[0])



rratcomp_arr = []
rsmacomp_arr = []
pericomp_arr = []


for sys in s:
    if float(sys.Rp/sys.Rs) <= 1: 
        rratcomp_arr.append(float(sys.Rp/sys.Rs))
        rsmacomp_arr.append(float((sys.Rp + sys.Rs)/sys.a ))
        pericomp_arr.append(float(sys.P))
    


    
pars1 = power_law(rratcomp_arr)
fit1 = sy.stats.beta.rvs(*pars1, size=len(rratcomp_arr))
pars2 = power_law(rsmacomp_arr)
fit2 = sy.stats.beta.rvs(*pars2, size=len(rsmacomp_arr))
pars3 = power_law(pericomp_arr)
fit3 = sy.stats.beta.rvs(*pars3, size=len(pericomp_arr))


plt.rcParams["figure.figsize"] = (10,18)
plt.rcParams["font.size"]=25

plt.suptitle("Radii selection: " + rtype[1] +  "  Data Set Size: " + str(len(s)))
plt.subplot(3,1,1)
plt.hist([rratcomp_arr,fit1])
plt.legend(["Model Data","Fit" ])
plt.title("Rratcomp Model and Estimates")
plt.xlabel("rratcomp")
plt.ylabel("Number of systems")


plt.subplot(3,1,2)
plt.hist([rsmacomp_arr,fit2])
plt.legend(["Model Data","Fit" ])
plt.title("Rsmacomp Model and Estimates")
plt.xlabel("rsmacomp")
plt.ylabel("Number of systems")


plt.subplot(3,1,3)
plt.hist([pericomp_arr,fit3])
plt.legend(["Model Data","Fit" ])
plt.title("Pericomp Model and Estimates")
plt.xlabel("pericomp")
plt.ylabel("Number of systems")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.4)
plt.show()

pars = [pars1, pars2, pars3]
par_arrs = [pericomp_arr, rsmacomp_arr, rratcomp_arr]
