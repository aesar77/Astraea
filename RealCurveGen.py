#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:53:42 2022

@author: lunaeaqua
"""
import time
import ciao_contrib.runtool as rt
from os import listdir, walk, makedirs, getcwd, chdir
from os.path import exists, dirname, exists, realpath
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as cu
import astropy.constants as cons
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams["font.size"]=12


class Real_Test_Curves():
    
    
    def __init__(self, ID, rate, errate, crvnum):
        self.ID = ID
        self.rate = rate
        self.errate = errate
        self.crvnum = crvnum
        self.dip = None
        self.diperr = None
    def set_dip(self,dip,diperr, points,delta):
        self.dip = dip
        self.diperr = diperr
        self.dip_points = points
        self.delta = delta        


def export_curves(trgts):
    whole_set = create_curve_set()
    req_obs = [o for o in whole_set if o.ID in trgts]
    return req_obs
        
def create_curve_set():       
    curve_set=[]
    
    main_path = "/home/lunaeaqua/Desktop/Python/CHANDRA_DATA/"
    paths = [x[0] for x in walk(main_path) if "Lightcurves" in x[0]]
    for path in paths:   
    
        rates= []
        erates= []
        rt= [x for x in listdir(path) if "CurveRate" in x]
        ert = [y for y in listdir(path) if "CurveErrRate" in y]
        
        target = rt[0].split("_")[1]
        
        rat = open(path +"/"+ rt[0], "r").read().split("]")
    
        for r in rat:
    
            rates.append([float(f) for f in r.strip("\n").strip("[").split(" ") if f !='' and  f != "nan"])
                
        errat = open(path +"/"+ ert[0], "r").read().split("]")
        for e in errat:
            erates.append([float(g) for g in e.strip("\n").strip("[").split(" ") if g !=''and  g != "nan"])
        crvnum = np.size(rates)
        
        curve_set.append(Real_Test_Curves(target, np.asarray(rates[:-1], dtype=(object)), np.asarray(erates[:-1], dtype=(object)), crvnum))
     
    for obs in curve_set:
        dips = []
        diperrs = []
        points = []
        deltas = []
        i = 0
        for cur in obs.rate:
            cur = np.asarray(cur)
        #dip parameters:
            ercur = obs.errate[i]
            Rs = np.random.randint(473, 964) * cons.R_sun/cu.m /1000
            print("Rs :", Rs)
            Rp = np.random.randint(5, 238) * cons.R_sun/cu.m /1000
            print("Rp :", Rp)
            delta = (Rp/Rs)**2 #(Rp/Rs)^2, fractional depth
            # # p = np.random.randint(38800,95.6*(10**6))/(300*1000)#orbital period [ks]
            # # print("p :", p)
            # a = np.random.randint(2, 17 ) /10 #semi-major axis [au]
            # # print("a :", a)
            # p = a**(3/2) /(300*1000)
            # T = int((Rs*p) / (np.pi*a)) #transit duration [ks]
            lngth = np.size(cur)
            dip_point = np.where(cur == min(cur))[0][0]
            T = np.random.randint(low=lngth*0.1, high=0.3*lngth)
            allrhs =  abs(dip_point-lngth)
            alllhs = abs(dip_point)
            print("dip pnt: ", dip_point, "allowed space on rhs: ", allrhs)
            print("dip pnt: ", dip_point, "allowed space on lhs: ", alllhs)
            print("T neeeded" , T)
            if allrhs < alllhs:
                rgt =  np.random.randint(dip_point, lngth)
                lft = rgt - T
            else:
                lft =  np.random.randint(0, dip_point + 1)
                rgt = lft + T
            dip_mean = abs(np.asarray(cur).mean() - delta)

            point = [lft,rgt]
            dip = np.random.poisson(dip_mean, T)
            print(len(dip))
            print(len(cur))
            dip_err = np.sqrt(dip)
            cur[lft:rgt] = cur[lft:rgt]- dip
            ercur[lft:rgt] = ercur[lft:rgt]- dip_err
            dips.append(cur)
            diperrs.append(ercur)
            points.append(point)
            deltas.append(delta)
            i += 1
        obs.set_dip(dips,diperrs,points,deltas)
    return curve_set
# a = export_curves(["RWAur"])
# for obs in a:
#     for j in range(0, len(obs.dip)):
#         plt.suptitle("Target: "+ obs.ID + ",  Curve no: " + str(j+1)+ ",  $\delta= $" +str(obs.delta[j]))   
#         plt.subplot(3,1,1)
#         plt.plot(obs.dip_points[j][0], obs.dip[j][obs.dip_points[j][0]], marker="X", color="red", markersize=12)
#         plt.plot(obs.dip_points[j][1], obs.dip[j][obs.dip_points[j][1]], marker="X", color="red", markersize=12)
#         plt.errorbar(np.arange(np.size(obs.dip[j])),obs.dip[j],yerr=obs.diperr[j], marker="o", linewidth=0,elinewidth=1, mfc="black",mec="black", ecolor="green", capsize=2)
#         plt.ylabel("Counts")
#         plt.subplot(3,1,2)
#         plt.errorbar(np.arange(np.size(obs.rate[j])),obs.rate[j],yerr=obs.errate[j], marker="o", linewidth=0,elinewidth=1, mfc="black",mec="black", ecolor="green", capsize=2)
#         plt.ylabel("Counts")
#         plt.xlabel("Time [ks]")
#         plt.subplot(3,1,3)
#         plt.plot(obs.dip_points[j][0], obs.dip[j][obs.dip_points[j][0]], marker="X", color="red", markersize=12)
#         plt.plot(obs.dip_points[j][1], obs.dip[j][obs.dip_points[j][1]], marker="X", color="red", markersize=12)
#         plt.errorbar(np.arange(np.size(obs.dip[j])),obs.dip[j],yerr=obs.diperr[j], marker="o", linewidth=0,elinewidth=1, mfc="black",mec="red", ecolor="green", capsize=2)
#         plt.errorbar(np.arange(np.size(obs.rate[j])),obs.rate[j],yerr=obs.errate[j], marker="o", linewidth=0,elinewidth=1, mfc="black",mec="black", ecolor="green", capsize=2)
#         plt.ylabel("Counts")
        
        # plt.show()

    


            
        