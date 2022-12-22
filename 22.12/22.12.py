#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:31:08 2022

@author: lunaeaqua
"""
#CURVE GENERATOR VER. 4:
#This version uses [a,b] sets for beta funct fitted to the model system parameter distribution. 
 
from ephesus import retr_rflxtranmodl, retr_indxtimetran
import numpy as np
import scipy as sc
from os import listdir, walk, makedirs, getcwd, chdir
from os.path import exists, dirname, exists, realpath
from sklearn.metrics import mean_squared_error
from ModelSystems_v2 import import_system, pars
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn import linear_model 
import time as t
start = t.time()

main_path = "/home/lunaeaqua/Desktop/Python/CHANDRA_DATA/ExtractedLCs"

curve_set=[] 
rates= []
erates= []
rratcomps = []
pericomps = []
epoccomps = []
rsmacomps = []
listepocs = []

# This part reads previously extracted light curves of a source. M51 OBsID in this particular case, bining = 1000
rt= [x for x in listdir(main_path) if "CurveRate" in x]
ert = [y for y in listdir(main_path) if "CurveErrRate" in y]

target = rt[0].split("_")[1]

rat = open(main_path +"/"+ rt[0], "r").read().split("]")
for r in rat:

    rates.append([float(f) for f in r.strip("\n").strip("[").split(" ") if f !='' and  f != "nan"][3:-3])
        
errat = open(main_path +"/"+ ert[0], "r").read().split("]")
for e in errat:
    erates.append([float(g) for g in e.strip("\n").strip("[").split(" ") if g !=''and  g != "nan"][3:-3])

Obs_13814R = [rates[3], erates[3]]
rmss = []

class Test_Curves():
    def __init__(self,rratcomp,cosicomp, pericomp, time, binnedtime, epoccomp,rsmacomp,crvnum, temp_curve, noisy_curve_norm, noisy_error_norm,transit, listepoch, duratrantotl):
        self.rratcomp = rratcomp
        self.cosicomp = cosicomp
        self.pericomp = pericomp
        self.time = time
        self.binnedtime = binnedtime
        self.epoccomp = epoccomp
        self.rsmacomp = rsmacomp
        self.crvnum = crvnum
        self.originalcrv = temp_curve
        self.noisycrv = noisy_curve_norm
        self.noisyerr = noisy_error_norm
        self.dipCenter = None
        self.matches = None
        self.width = None
        self.dipMean = None
        self.outsideDipMean = None
        self.nbPairs = None
        self.conditions = None
        self.params = None
        self.transit = transit #True if the curve includes transit, False if it does not.
        self.result = False
        self.info = ""
        self.binaryR = 0
        self.ratio = None
        self.listepoch = listepoch
        self.epoch = None
        self.correspondance = None
        self.durtran = duratrantotl
        if self.transit == True:
            self.binaryTr = 1
        if self.transit ==False:
            self.binaryTr = 0
        
    def set_parameters(self, dip_mean, dip_extracted_mean, nb_pairs, dip_center, match, width, conditions, params, epoch):
        self.dipCenter = dip_center
        self.matches = match
        self.width = width
        self.dipMean = dip_mean
        self.outsideDipMean = dip_extracted_mean
        self.nbPairs = nb_pairs
        self.conditions = conditions
        self.params = params
        self.ratio = dip_mean/dip_extracted_mean
        self.epoch = epoch
    
    def set_correspondance(self, corr):
        self.correspondance = corr
        
    def set_result(self, r):
        self.result = r
        if r == True:
            self.binaryR = 1

        
    def set_info(self, i):
        self.info = self.info + "\n" + i
        
    def get_info(self):
        print(self.info)
        
        
def normalise(matrix):
    norm = np.linalg.norm(matrix)
    normalised = matrix/norm
    return normalised

def rms_values(obs, sim):
    rms = mean_squared_error(obs, sim, squared = False)
    rmss.append('{:.2f}%'.format(rms))
    return '{:.2f}%'.format(rms)

def bins(curve, bining, time):
       
    bin_num = np.size(time)//bining
    binned_crv =  np.zeros(bin_num)
    binned_time = np.linspace(0, max(time), bin_num)
    
    #divding the curve into bins, filling up each bin with corresonding value on the curve
    for j in range(0,bin_num):
    
        binned_crv[j] = sum(curve[bining*j:bining*(j+1)])
                         
    return binned_crv, binned_time

def extract_transits(curve):
    epos = []
    epoend = 0
    period =  curve.pericomp[0] * 86400 
    
    epostart = curve.listepoch[0] 
    epoend = epostart + curve.durtran[0]*3600
    epos.append([int(epostart), int(epoend)])
    
    while  epoend + period < curve.time[-1] and len(epos) < 3:
        epostart = curve.listepoch[0] + period
        epoend = epostart + curve.durtran[0]*3600
        epos.append([int(epostart), int(epoend)])
        
    return epos  
        
def import_test_curveset(crvnum, band, bining):

    crv_set = []
       
    for i in range(0, crvnum):


        
        min_tr_dur = 80000
        max_tr_dur = 100000
        time = np.arange(start=0,stop=np.random.randint(min_tr_dur,max_tr_dur), step= 1) 
        lenT = len(time)     
        crvnum = i + 1
        
        rratpars, rsmapars, peripars = pars
        
        rratcomp = [sc.stats.beta.rvs(*rratpars, size=1)][0] #rplanet/rstar
        cosicomp = [0]
        pericomp = [sc.stats.beta.rvs(*peripars, size=1)][0]#scipy ranfom distributions check.
        epoccomp = [lenT/3 + np.random.rand()*lenT/3]
        rsmacomp =  [sc.stats.beta.rvs(*rsmapars, size=1)][0]
        
        rratcomps.append(rratcomp)
        pericomps.append(pericomp)
        epoccomps.append(epoccomp)
        rsmacomps.append(rsmacomp)
        
        get_curve = retr_rflxtranmodl(time=time/86400, pericomp=pericomp, epocmtracomp=epoccomp, rratcomp=rratcomp, rsmacomp=rsmacomp, cosicomp = cosicomp, typenorm = 'maxm')
        duratran = get_curve["duratrantotl"]*3.6
        temp_curve = get_curve["rflx"] #temp_curve = get_curve["dur"] #rtetr_retrieve["durtat"] ephesus.retr_duratotl() 
        listepoch =  retr_indxtimetran(time/86400, epoccomp[0], pericomp[0], duratran)
        original_binned = bins(temp_curve, bining, time)[0] * band
        
        if len(listepoch) == 0:
            transit = False
        else:
            transit = True
        
        s = temp_curve * band
      
        noise_curve = sc.stats.poisson.rvs(abs(s))
        binnedcrv, binnedtime = bins(noise_curve, bining, time)
        noisy_curve_norm = (binnedcrv)
        noisy_error_norm = np.sqrt(noisy_curve_norm)
        crv_set.append(Test_Curves(rratcomp, cosicomp, pericomp, time, binnedtime, epoccomp, rsmacomp, crvnum, original_binned, noisy_curve_norm, noisy_error_norm,transit, listepoch, duratran))    
       

        listepocs.append(listepoch)


    return  crv_set



    
def detect_dip(curve, width, matching, outside_dip_mean, dip_mean):
    pa = 7200 #in seconds, taken from nasa website (...transits last 2 to 16 hours)
    pb = 2
    pc = 1
    
    if width>= pa:
        a =  True
    else:
        a =  False
    
    if sum(matching) >= pb:       
        b =  True
    else:
        b =  False
        
    if dip_mean/outside_dip_mean <= pc:
        c =  True
    else:
        c =  False
        
     
    return a, b, c, pa, pb, pc


#Energy band count numbers derivedby the fluxes of M51 
ultrasoft = 0.0016, "Ultrasoft" 
soft = 0.00904 , "Soft"
medium = 0.000858, "Medium"
hard = 7.46e-5, "Hard"
broad = 0.00968, "Broad"
merged_broad = 0.0063, "Merged broad"

bining = 1000
bands = [ultrasoft, soft, medium, hard, broad, merged_broad]
band = bands[5]
tr = import_test_curveset(100, band[0], bining)
curves = tr 
i=0
    
paramc1 = []
paramc2 = []
paramc3 = []
paramc4 = []
total_dip_found = []
plt.rcParams["font.size"]=30
for c in curves:
    curve = c.noisycrv
    error = c.noisyerr
    orcurve = c.originalcrv
    ortime = c.time
    time = c.binnedtime
    if c.transit == True:
        epoch = extract_transits(c)
    else:
        epoch = [[]]
    rms = rms_values(orcurve, curve)
    
    mean = curve.mean()
    dip_points = np.where(curve == min(curve))[0] #all points that hit the minimum
    
    widths= []
    nb_pairs = []
    
    for dip in dip_points:
        
        
        rnb = curve[dip]
        lnb = curve[dip]
        tolerance = abs(mean - curve[dip])/np.sqrt(2)
        
        i = 0
        while (abs(rnb - curve[dip + i]))/np.sqrt(2) < tolerance and dip+i < len(curve)-1 :
            
            try:                
                rnb = curve[dip + i]
                i += 1
                
            except:                        
                break
            
        j = 0                
        while (abs(lnb - curve[dip - j]))/np.sqrt(2)  < tolerance and dip-j>=0:
            
            try:                 
                lnb = curve[dip - j]
                j += 1   
                
            except:                        
                break
        width = i+j
        nb_pairs.append([dip - j , dip + i])
        widths.append(width)
        i = 0
        j = 0
                
    if np.size(widths) == 0 or widths[0] < 2:
        print("no suitable minima for a dip candidate:" + str(c.crvnum))
        curves.remove(c)
        #widths.remove(width)
        #nb_pairs.remove(nb_pairs[-1])
        
    else:
        dip_center = dip_points[widths.index(max(widths))] #in case of more than one minima, the widest one is chosen as the minima at the dip
        pairs = nb_pairs[np.where(dip_points == dip_center)[0][0]] #boundaries of the dip where the chosen minima above included
        match = []
        pair_means = []
        
        dip_mean = curve[min(pairs)+1 : max(pairs)].mean()
        dip_extracted_mean = np.delete(curve, np.arange(min(pairs)+1, max(pairs)-1)).mean() #the mean of the curve without the dip
    
       
        for k in range(1,5):
            
            try:
                cl = curve[pairs[0] - k]
                cr = curve[pairs[1] + k]
                sigma = np.sqrt(max(cl, cr))
                
                
                if abs(cr-cl) < 2* sigma:
                    match.append(1)
                else:
                    match.append(0)
                    
                    
            except:
                break
            
            
      
        cond1, cond2, cond3, par1, par2, par3  = detect_dip(curve, max(widths)*bining, match, dip_extracted_mean, dip_mean)
        
        conditions = [cond1, cond2, cond3]
        local_params = [" >= " + str(par1), " >= " + str(par2), " <= "+ str(par3)]
        c.set_parameters(dip_mean, dip_extracted_mean, nb_pairs, dip_center, match, max(widths)*bining, conditions, local_params, epoch)
        if len(epoch[0]) !=0:
            c.set_correspondance( -(abs(pairs[0] - c.epoch[0][0]) + abs(pairs[1] - c.epoch[0][1]))/1000)
        paramc1.append((max(widths) - 2)*bining)
        paramc2.append(sum(match))
        paramc3.append(dip_mean)
        paramc4.append(dip_extracted_mean)
        c.set_info("Curve number: " + str(c.crvnum))
        c.set_info("Expected dip(s) between [seconds]: " + str(c.epoch))
        #c.set_info("Expected dip between: " + str([ortime[np.where(orcurve != band)[0][0]],ortime[np.where(orcurve != band)[0][-1]]]) + "[d]")
        c.set_info("C1 parameter: width/total length = " + str(c.width/max(c.time)) + c.params[0] + "\nC2 parameter: matches: " + str(sum(c.matches)) + c.params[1] + "\nC3 parameter: dip mean/outside dip mean: " + str(c.dipMean/c.outsideDipMean) + str(c.params[2]))
        
        for cnd in range(0,3):
            
            if conditions[cnd] == True:
                c.set_info("Condition " + str(cnd+1) + " checked" )
                
            else:
                c.set_info("Condition " + str(cnd+1) + " failed" )
    
                
        if False not in conditions:
            c.set_info("Result: Dip found:")
            c.set_info("Approximate eclipse duration [days]: " + str([time[pairs[1]]-time[pairs[0]]]))
            c.set_info("Possible dip between [seconds] :" + str([time[pairs[0]], time[pairs[1]]]) + "\n")
            c.set_result(True)
            total_dip_found.append(1)
            
        else:
            c.set_info("Searched for a dip between :" + str([time[pairs[0]], time[pairs[1]]]) + "\n")
            c.set_info("Result: No dip found.\n")    
            total_dip_found.append(0)
            

        if c.transit == True:
            plt.rcParams["figure.figsize"] = (30,10)
            plt.suptitle("No " + str(c.crvnum) + "  Transit : " + str(c.transit) + "  Result=" + str(c.result) + ",  Bin Size="+str(bining) + ',  RMS=' + str(rms))
            plt.subplot(1,1,1).set_title("M51 Light Curves")
            plt.ylabel("Counts per bin")
    
            plt.errorbar(time/bining,curve,yerr = error, marker="o", linewidth=0,elinewidth=0.8, mfc="black",mec="black", ecolor="black", capsize=3)
            plt.errorbar(np.linspace(0, len(time), 179), Obs_13814R[0], yerr = Obs_13814R[1], marker="o", linewidth=0,elinewidth=0.5, mfc="blue",mec="blue", ecolor="blue", capsize=3)
            plt.plot(time/bining, orcurve, linewidth=6, color ="red")
            if c.result == True:
                plt.axvline(time[pairs[0]]/bining, ymin=0,ymax=max(curve), linestyle="--", color="g")
                if len(epoch[0]) !=0:
                    plt.axvline(c.epoch[0][0]/bining, ymin=0,ymax=max(curve), linestyle="--", color="r")
                plt.legend(["Simulated Curve without noise","Detected Transit Interval","Simulated Transit Interval","Simulated Curve","Obs 13814"])  
                plt.axvline(time[pairs[1]]/bining, ymin=0,ymax=max(curve), linestyle="--", color="g")
                if len(epoch[0]) !=0:
                    plt.axvline(c.epoch[0][1]/bining, ymin=0,ymax=max(curve), linestyle="--", color="r")
            else:
                if len(epoch[0]) !=0:
                    plt.axvline(c.epoch[0][0]/bining, ymin=0,ymax=max(curve), linestyle="--", color="r")
                plt.legend(["Simulated Curve without noise","Simulated Transit Interval","Simulated Curve","Obs 13814"])  
                if len(epoch[0]) !=0:
                    plt.axvline(c.epoch[0][1]/bining, ymin=0,ymax=max(curve), linestyle="--", color="r")
                
            # plt.plot(ortime[np.where(orcurve != ultrasoft)[0][0]]/500, orcurve[np.where(orcurve != ultrasoft)[0][0]], marker="v", markersize=15, color="blue", linewidth=0)
            # plt.plot(ortime[np.where(orcurve != ultrasoft)[0][-1]]/500, orcurve[np.where(orcurve != ultrasoft)[0][-1]], marker="v", markersize=15, color="blue", linewidth=0)
    
            plt.plot(time/bining, orcurve)
            #plt.subplot(2,1,2).set_title("Simulated LC")
            
            plt.ylabel("Counts per bin")
            plt.xlabel("Time [ks]")
            plt.show()


#STATS
transit_curves = np.array([c for c in curves if c.transit == True])
nontransit_curves = np.array([c for c in curves if c.transit == False])

plt.rcParams["font.size"]=30
#[c.get_info() for c in curves]


dips_found = sum(total_dip_found)
undetected_signal = np.size([c for c in curves if c.transit == True and c.result == False])
true_signal = np.size([c for c in curves if c.transit == True and c.result == True])
false_signal = np.size([c for c in curves if c.transit == False and c.result == True])


print("[ver1] At total ",dips_found , " transits found among " , str(len(curves)) , "simulated curves\n")
print("Expected dip number :", len([c for c in curves if c.transit == True]))
print("Detected true signals: ", true_signal)
print("Detected false signals: ", false_signal)
print("Undetected signals: ", undetected_signal)
print("Runtime [s]: ", ("%s seconds" % (t.time() - start)))

plt.rcParams["figure.figsize"] = (30,15)
plt.subplot(2,2,1)
plt.hist([np.array(paramc1)/np.array([max(x.time) for x in curves]), np.array([x.width for x in curves if x.result == True])/np.array([max(x.time) for x in curves if x.result == True]), np.array([x.width for x in curves if x.result == False])/np.array([max(x.time) for x in curves if x.result==False])],bins=np.arange(0.05,1.1,0.1))
plt.title("Transit Depth / Total Time")
plt.legend(["All curves", "Curves Detected", "Curves Undetected"])
plt.subplot(2,2,2)
plt.hist([paramc2, [sum(c.matches) for c in curves if c.result == True], [sum(c.matches) for c in curves if c.result == False]], bins=np.arange(0.5,6,1))
plt.legend(["All curves", "Curves Detected", "Curves Undetected"])
plt.title("Number of Matching Pairs")
plt.subplot(2,2,3)
plt.hist([np.asarray(paramc3)/np.asarray(paramc4), np.array([c.dipMean for c in curves if c.result == True])/np.array([c.outsideDipMean for c in curves if c.result == True]), np.array([c.dipMean for c in curves if c.result == False])/np.array([c.outsideDipMean for c in curves if c.result == False])],bins=np.arange(0.05,1.1,0.1))
plt.legend(["All curves", "Curves Detected", "Curves Undetected"])
plt.title("Transit Depth")
plt.subplot(2,2,4)
plt.hist([np.asarray(paramc4) - np.asarray(paramc3), np.array([c.outsideDipMean for c in curves if c.result == True])-np.array([c.dipMean for c in curves if c.result == True]), np.array([c.outsideDipMean for c in curves if c.result == False]) - np.array([c.dipMean for c in curves if c.result == False])])
plt.legend(["All curves", "Curves Detected", "Curves Undetected"])
plt.title("Transit Depth Difference (Outside Dip Means - Dip Means)")
plt.show()

plt.rcParams["figure.figsize"] = (30,15)
plt.subplot(2,2,1)
plt.hist([np.array([x.rratcomp[0] for x in curves if x.result == True]), np.array([x.rratcomp[0] for x in curves if x.result == False])])
plt.title("rratcomp")
plt.legend(["Curves Detected", "Curves Undetected"])
plt.subplot(2,2,2)
plt.hist([np.array([x.pericomp[0] for x in curves if x.result == True]), np.array([x.pericomp[0] for x in curves if x.result == False])])
plt.title("pericomp")
plt.legend(["Curves Detected", "Curves Undetected"])
plt.subplot(2,2,3)
plt.hist([np.array([x.rsmacomp[0] for x in curves if x.result == True]), np.array([x.rsmacomp[0] for x in curves if x.result == False])])
plt.title("rsmacomp")
plt.legend(["Curves Detected", "Curves Undetected"])
plt.subplot(2,2,4)
plt.hist([np.array([x.epoccomp[0]/len(x.time) for x in curves if x.result == True]), np.array([x.epoccomp[0]/len(x.time) for x in curves if x.result == False])])
plt.title("epoccomp")
plt.legend(["Curves Detected", "Curves Undetected"])
plt.show()

plt.rcParams["figure.figsize"] = (25,15)
plt.suptitle("Recall with respect to rflxmodl parameters")
plt.subplot(2,2,1)
rrath = np.histogram(np.array([x.rratcomp[0] for x in curves if x.result == True]))
rrathf = np.histogram(np.array([x.rratcomp[0] for x in curves if x.result == False]))
plt.scatter(rrath[1][:-1], rrath[0]/(rrath[0]+rrathf[0]),s=100)
plt.title("rratcomp")
plt.subplot(2,2,2)
perih = np.histogram(np.array([x.pericomp[0] for x in curves if x.result == True]))
perihf = np.histogram(np.array([x.pericomp[0] for x in curves if x.result == False]))
plt.scatter(perih[1][:-1], perih[0]/(perih[0]+perihf[0]),s=100)
plt.title("pericomp")

plt.subplot(2,2,3)
rsmah = np.histogram(np.array([x.rsmacomp[0] for x in curves if x.result == True]))
rsmahf = np.histogram(np.array([x.rsmacomp[0] for x in curves if x.result == False]))
plt.scatter(rsmah[1][:-1], rsmah[0]/(rsmah[0]+rsmahf[0]),s=100)
plt.title("rsmacomp")

plt.subplot(2,2,4)
plt.title("epoccomp")
epoch = np.histogram(np.array([x.epoccomp[0] for x in curves if x.result == True]))
epochf = np.histogram(np.array([x.epoccomp[0] for x in curves if x.result == False]))
plt.scatter(epochf[1][:-1], epoch[0]/(epoch[0]+epochf[0]),s=100)
plt.show()



plt.rcParams["figure.figsize"] = (10,18)

plt.suptitle("Recall")
plt.subplot(2,1,1)
plt.title("Total Dips Expected")
yyy = [true_signal, undetected_signal]
plt.pie(yyy, labels=["Detected Dips", "Undetected Dips"], autopct='%1.1f%%')
if undetected_signal !=0:
    plt.subplot(2,1,2)
    plt.title("Failed Conditions")
    yyyy = [np.size([c for c in curves if c.conditions[0]==False]), np.size([c for c in curves if c.conditions[1]==False]), np.size([c for c in curves if c.conditions[2]==False]) ]
    plt.pie(yyyy, labels=["Condition 1", "Condition 2", "Condition 3"], autopct='%1.1f%%')
    plt.show()



plt.rcParams["figure.figsize"] = (20,15)
plt.subplot(3,1,1)
for w in curves:
    if w.result == False:
        plt.scatter(curves.index(w), w.width, color = "red", s=80)
        
    if w.result == True:
        plt.scatter(curves.index(w), w.width, color = "green", s=80)
        
        
plt.hlines(par1,linestyles="-", xmin=0, xmax=len(curves))

plt.ylabel("Transit Depth")          

legend_elements = [Line2D([0], [0], marker='o', color='r',lw=0, label='Dip negative'),
                   Line2D([0], [0], marker='o', color='g',lw=0, label='Dip positive'),
                   Line2D([0], [0], lw = 3 , color='b', label='Parameter value')]
plt.legend(handles=legend_elements)
plt.subplot(3,1,2)
for w in curves:
    if w.result == False:
        plt.scatter(curves.index(w), sum(w.matches), color = "red", s=80)
        
    if w.result == True:
        plt.scatter(curves.index(w), sum(w.matches), color = "green",s=80)

plt.hlines(par2,linestyles="-", xmin=0, xmax=len(curves))
plt.ylabel("Matches")               
plt.subplot(3,1,3)
for w in curves:
    if w.result == False:
        plt.scatter(curves.index(w), w.dipMean/w.outsideDipMean , color = "red",s=80)
        
    if w.result == True:
        plt.scatter(curves.index(w), w.dipMean/w.outsideDipMean, color = "green",s=80)
        
plt.hlines(par3,linestyles="-", xmin=0, xmax=len(curves))
plt.ylabel("Transit Depth")
plt.xlabel("Curve Number")             
plt.show()

plt.stem([c.correspondance for c in curves])
plt.title(" Detected dip - Actual dip indices (0 for exact correspondance) ")
plt.ylabel("Indice Correspondance")
plt.xlabel("Curve Number")    
plt.show()

plt.rcParams["figure.figsize"] = (25,30)
plt.suptitle("Recall / Parameters")
plt.subplot(4,1,1)
sorted_rratcomp = sorted(tr, key = lambda x: x.rratcomp)
plt.plot([c.rratcomp for c in sorted_rratcomp], [c.binaryR for c in sorted_rratcomp], linewidth=0, marker="o", markersize=10)
plt.xlabel("rratcomp")
plt.subplot(4,1,2)
sorted_pericomp = sorted(tr, key = lambda x: x.pericomp)
plt.plot([c.pericomp for c in sorted_pericomp], [c.binaryR for c in sorted_pericomp], linewidth=0, marker="o",  markersize=10)
plt.xlabel("pericomp")
plt.subplot(4,1,3)
sorted_rsmacomp = sorted(tr, key = lambda x: x.rsmacomp)
plt.plot([c.rsmacomp for c in sorted_rsmacomp], [c.binaryR for c in sorted_rsmacomp], linewidth=0, marker="o", markersize=10)
plt.xlabel("rsmacomp")
plt.subplot(4,1,4)
sorted_epocomp = sorted(tr, key = lambda x: x.epoccomp)
plt.plot([c.epoccomp for c in sorted_epocomp], [c.binaryR for c in sorted_epocomp], linewidth=0, marker="o", markersize=10)
plt.xlabel("epoccomp")
plt.show()


# sorted_width = sorted(curves, key=lambda x: x.width)
# # sorted_match = sorted(curves, key=lambda x: x.match)
# sorted_ratio = sorted(curves, key=lambda x: x.ratio)
# W = np.array(sorted_width)
# #
# R = np.array(sorted_ratio)
# logrw = linear_model.LogisticRegression()
# logrr = linear_model.LogisticRegression()
# Wf = logrw.fit(np.array([c.width for c in W]).reshape(-1, 1), np.array([c.binaryR for c in W]))
# # Mf = logr.fit(np.array([c.match for c in M]).reshape(-1, 1), np.array([c.binaryR for c in M]))
# Rf = logrr.fit(np.array([c.ratio for c in R]).reshape(-1, 1), np.array([c.binaryR for c in R]))

# def logit2prob(logr, X):
#   log_odds = logr.coef_ * X + logr.intercept_
#   odds = np.exp(log_odds)
#   probability = odds / (1 + odds)
#   return(probability)

#print(logit2prob(logrw, np.array([c.width for c in W]).reshape(-1, 1)))

# plt.subplot(2,1,1)
# plt.plot(np.array([c.width for c in W]),logit2prob(logrw, np.array([c.width for c in W]).reshape(-1, 1)))
# plt.ylabel("Recall")
# plt.xlabel("Transit duration")
# plt.subplot(3,1,2)
# plt.plot(logit2prob(logr, np.array([c.match for c in M]).reshape(-1, 1)))
# plt.xlabel("match")
# plt.subplot(2,1,2)
# plt.plot(np.array([c.ratio for c in R]), logit2prob(logrr, np.array([c.ratio for c in R]).reshape(-1, 1)))
# plt.xlabel("Transit depth")
# plt.ylabel("Recall")
# plt.show()

# plt.scatter(W,binary, color = "green", s=80)
# plt.plot(logit2prob(logr, W))        
        
# plt.hlines(par1,linestyles="-", xmin=0, xmax=len(curves))

# plt.ylabel("Width")          

# legend_elements = [Line2D([0], [0], marker='o', color='r',lw=0, label='Dip negative'),
#                    Line2D([0], [0], marker='o', color='g',lw=0, label='Dip positive'),
#                   # Line2D([0], [0], marker='x', color='y',lw=0, label='Dip detected'),
#                    Line2D([0], [0], lw = 3 , color='b', label='Parameter value')]
# plt.legend(handles=legend_elements)
# plt.subplot(3,1,2)
# for w in curves:
#     if w.result == False:
#         plt.scatter(sum(w.matches),binary, color = "red", s=80)
        
#     if w.result == True:
#         plt.scatter(sum(w.matches),binary, color = "green",s=80)

# plt.hlines(par2,linestyles="-", xmin=0, xmax=len(curves))
# plt.ylabel("Matches")               
# plt.subplot(3,1,3)
# for w in curves:
#     if w.result == False:
#         plt.scatter((w.dipMean/w.outsideDipMean) ,binary, color = "red",s=80)
        
#     if w.result == True:
#         plt.scatter((w.dipMean/w.outsideDipMean), binary, color = "green",s=80)
        
# plt.hlines(par3,linestyles="-", xmin=0, xmax=len(curves))
# plt.ylabel("Dip Mean/Outside Dip Mean")
# plt.xlabel("Curve Number")             
# plt.show()


