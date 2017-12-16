import FitterFunctions
import numpy as np
from scipy.linalg import pinv2
from scipy.integrate import quad
import pathos.multiprocessing as mp
from iminuit import Minuit
from numba import jit
import time
import mpmath
mp.dps = 1000000
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

NumberOfSamples = 1000

NearDet = True

#Live-time
LivetimeRatio = .9567
NewFarLivetimeRatio = .9560
NewNearLivetimeRatio = .7900

#Misc 
Bugey4Fractions = np.array([0.538, 0.078, 0.328, 0.056])
M13Mean = 2.44e-3
M13LeftSigma = .1e-3
M13RightSigma = .09e-3
Li9RateMean = 0.87
Li9RateSigma = .42
NearLi9RateMean = 4.67
NearLi9Sigma = 1.42
FNSMRateMean = 0.47
FNSMRateSigma = 0.03
NearFNSMRateMean = 4.
NearFNSMRateSigma = 0.15
NearAccMean = .303
NearAccSigma = 0.007
SSTT13Mean = .09


RootPath = "/Users/dhooghe/Desktop/Work/Analysis"
print "Getting Binning"
execfile(RootPath+"/Bins.py")

#Data
OldFarData = np.load(RootPath+"/Data/OldFDEvents.npy")
FarData = np.load(RootPath+"/Data/FDEvents.npy")
NearData = np.load(RootPath+"/Data/NDEvents.npy")
OldFarEventRunNumbers = np.load(RootPath+"/Data/OldFDRunNumbers.npy")
FarEventRunNumbers = np.load(RootPath+"/Data/FDRunNumbers.npy")
NearEventRunNumbers = np.load(RootPath+"/Data/NDRunNumbers.npy")

#Fluxes
Fluxes = np.load(RootPath+"/FluxGeneration/Fluxes.npy")

#Fissions
R1Fissions = np.load(RootPath+"/FissionGeneration/R1FarFissions.npy")
R2Fissions = np.load(RootPath+"/FissionGeneration/R2FarFissions.npy")
NewR1Fissions = np.load(RootPath+"/FissionGeneration/NewR1FarFissions.npy")
NewR2Fissions = np.load(RootPath+"/FissionGeneration/NewR2FarFissions.npy")
NearR1Fissions = np.load(RootPath+"/FissionGeneration/NewR1NearFissions.npy")
NearR2Fissions = np.load(RootPath+"/FissionGeneration/NewR2NearFissions.npy")

#Run Lengths
OldFarRuntimes = np.load(RootPath+"/RunInformationGeneration/ThirdPubRunTimes.npy")
FarRuntimes = np.load(RootPath+"/RunInformationGeneration/NewFarRunTimes.npy")
NearRuntimes = np.load(RootPath+"/RunInformationGeneration/NearRunTimes.npy")

#Energy Systematics
OldFarMean = np.load(RootPath+"/EnergySystematics/OldFarMean.npy")
FarMean = np.load(RootPath+"/EnergySystematics/FarMean.npy")
NearMean = np.load(RootPath+"/EnergySystematics/NearMean.npy")
OldFarCov = np.load(RootPath+"/EnergySystematics/OldFarCov.npy")
FarCov = np.load(RootPath+"/EnergySystematics/FarCov.npy")
NearCov = np.load(RootPath+"/EnergySystematics/NearCov.npy")

#Background Events
OldFarAccidentals = np.load(RootPath+"/BackgroundGeneration/GetAccidentals/OldFD_Accidentals.npy")
FarAccidentals = np.load(RootPath+"/BackgroundGeneration/GetAccidentals/FD_Accidentals.npy")
NearAccidentals = np.load(RootPath+"/BackgroundGeneration/GetAccidentals/ND_Accidentals.npy")
Li9Events = np.load(RootPath+"/BackgroundGeneration/GetLithium/Li9Events.npy")

#TrueToVisible
OldFar_True = np.load(RootPath+"/TrueToVisible/FDITrues.npy")
OldFar_Visible = np.load(RootPath+"/TrueToVisible/FDIVisibles.npy")
Far_True = np.load(RootPath+"/TrueToVisible/FDIITrues.npy")
Far_Visible = np.load(RootPath+"/TrueToVisible/FDIIVisibles.npy")
Near_True = np.load(RootPath+"/TrueToVisible/NDTrues.npy")
Near_Visible = np.load(RootPath+"/TrueToVisible/NDVisibles.npy")

#Corrections
NearCorrections = np.load(RootPath+ "/OffEquilibrium/NearCorrections.npy")
FarCorrections = np.load(RootPath+ "/OffEquilibrium/NewFarCorrections.npy")
OldFarCorrections = np.load(RootPath+ "/OffEquilibrium/FarCorrections.npy")

#Baselines
R1FarBaselines = np.load(RootPath+"/Baselines/FarR2Baselines.npy")
R2FarBaselines = np.load(RootPath+"/Baselines/FarR1Baselines.npy")
R1NearBaselines = np.load(RootPath+"/Baselines/NearR2Baselines.npy")
R2NearBaselines = np.load(RootPath+"/Baselines/NearR1Baselines.npy")

#Near Normalization
RunFractions = np.load(RootPath+"/NearNormalization/RunFractions.npy")
RunDelta = np.load(RootPath+"/NearNormalization/RunDelta.npy")

#Baseline Calculations
R1FarMean = np.mean(R1FarBaselines)
R2FarMean = np.mean(R2FarBaselines)
R1FarHistogram, R1FarEdges = np.histogram(R1FarBaselines, bins=500, density=True)
R2FarHistogram, R2FarEdges = np.histogram(R2FarBaselines, bins=500, density=True)
R1NearMean = np.mean(R1NearBaselines)
R2NearMean = np.mean(R2NearBaselines)
R1NearHistogram, R1NearEdges = np.histogram(R1NearBaselines, bins=500, density=True)
R2NearHistogram, R2NearEdges = np.histogram(R2NearBaselines, bins=500, density=True)
R1FarCenters = FitterFunctions.GetBinCenters(R1FarEdges)
R2FarCenters = FitterFunctions.GetBinCenters(R2FarEdges)
R1NearCenters = FitterFunctions.GetBinCenters(R1NearEdges)
R2NearCenters = FitterFunctions.GetBinCenters(R2NearEdges)
print "Checking normalization R1 and R2 Baseline Distributions (Far Then Near)"
print np.sum(R1FarHistogram)*(R1FarEdges[1]-R1FarEdges[0]), np.sum(R2FarHistogram)*(R2FarEdges[1]-R2FarEdges[0])
print np.sum(R1NearHistogram)*(R1NearEdges[1]-R1NearEdges[0]), np.sum(R2NearHistogram)*(R2NearEdges[1]-R2NearEdges[0])
print "Far Means: ", R1FarMean, R2FarMean
print "Near Means: ", R1NearMean, R2NearMean

#True To Visible Histogram Generation
FDI_Hs = FitterFunctions.GetTrueToVisibleConversion(NumberOfSamples, OldFar_True, OldFar_Visible, NeutrinoEnergyBinEdges, VisibleEnergyBinEdges)
FDII_Hs = FitterFunctions.GetTrueToVisibleConversion(NumberOfSamples, Far_True, Far_Visible, NeutrinoEnergyBinEdges, VisibleEnergyBinEdges)
ND_Hs = FitterFunctions.GetTrueToVisibleConversion(NumberOfSamples, Near_True, Near_Visible, NeutrinoEnergyBinEdges, VisibleEnergyBinEdges)

#ND Veto 
NDVeto = np.load(RootPath+"/FissionGeneration/NewNearVeto.npy")

#Background Histograms
Li9Histogram, a = np.histogram(Li9Events, bins=VisibleEnergyBinEdges, density=False)
FarAccHistogram, a = np.histogram(OldFarAccidentals, bins=VisibleEnergyBinEdges, density=False)
NearAccHistogram, a = np.histogram(FarAccidentals, bins=VisibleEnergyBinEdges, density=False)
OldFarAccHistogram, a = np.histogram(NearAccidentals, bins=VisibleEnergyBinEdges, density=False)

Li9Histograms, AccHistograms, NearAccHistograms, OldAccHistograms, FNHistograms = FitterFunctions.GetBackgroundHistograms(NumberOfSamples, a, Li9Histogram, OldFarAccHistogram, FarAccHistogram, NearAccHistogram)
print "Checking that background histograms are normalized:"
print np.sum(Li9Histograms[0])*dVisibleEnergy, np.sum(AccHistograms[0])*dVisibleEnergy, np.sum(FNHistograms[0])

#Inverse Energy Covariances
InvOldFarCov = pinv2(OldFarCov)
InvFarCov = pinv2(FarCov)
InvNearCov = pinv2(NearCov)

#Efficiences
OldFarEfficiencies, FarEfficiencies, NearEfficiencies = FitterFunctions.GetEfficiencies(NumberOfSamples)
print "Mean Third Pub Efficiency: ", np.mean(OldFarEfficiencies), ' pm ', np.std(OldFarEfficiencies)
print "Mean New Far Efficiency: ", np.mean(FarEfficiencies), ' pm ', np.std(FarEfficiencies)
print "Mean Near Efficiency: ", np.mean(NearEfficiencies), ' pm ', np.std(NearEfficiencies)

#Data Shapes
OldFarEventList = FitterFunctions.OrganizeEvents(OldFarData, OldFarEventRunNumbers, RootPath+"/Data/TAGGED_DC3rdPub_CTv5_ALL_v1.txt")
FarEventList = FitterFunctions.OrganizeEvents(FarData, FarEventRunNumbers, RootPath+"/Data/TAGGED_DCFifteenMonths_WithReactorData_FD_v2.txt")
NearEventList = FitterFunctions.OrganizeEvents(NearData, NearEventRunNumbers, RootPath+"/Data/TAGGED_DCFifteenMonths_WithReactorData_ND_v2.txt")
NearEventList, NearData = FitterFunctions.VetoEvents(NearEventList, NDVeto)

OldFarEventNumberList = FitterFunctions.GetNumberOfEventsForEachRun(OldFarEventList)
FarEventNumberList = FitterFunctions.GetNumberOfEventsForEachRun(FarEventList)
NearEventNumberList = FitterFunctions.GetNumberOfEventsForEachRun(NearEventList)

#Bump Samples
BumpSpectrum = np.load("/Users/dhooghe/Desktop/Work/Analysis/Beta/DwyerFlux.npy")

#Spectrum Functions
@jit
def Spectrum(i, Args):
    return quad(FitterFunctions.IntegratedSpectrumSansFlux, NeutrinoEnergyBinEdges[i], NeutrinoEnergyBinEdges[i+1], args=Args, epsrel = 1.e-10, epsabs = 0.)[0]
@jit
def NoOscSpectrum(i, Args):
    return quad(FitterFunctions.IntegratedUnOscillatedSpectrumSansFlux, NeutrinoEnergyBinEdges[i], NeutrinoEnergyBinEdges[i+1], args=Args, epsrel = 1.e-10, epsabs = 0.)[0]

Count = 0
OldBumpNorm = 0.
OldNearNorm_A = 0.
OldNearNorm_B = 0.
OldNearNorm_C = 0.
OldSSTT13 = 0.
OldM13 = 0.
OldLi9Rate = 0.
OldFNSMRate = 0.
OldNearLi9Rate = 0.
OldNearFNSMRate = 0.
OldBugey4Coeff = 0.
OldNear_A = 0.
OldNear_B = 0.
OldNear_C = 0.
OldFar_A = 0.
OldFar_B = 0.
OldFar_C = 0.
OldOldFar_A = 0.
OldOldFar_B = 0.
OldOldFar_C = 0.
NearNormalization = np.ones([RunFractions.size])
OldFarSpectra = []
FarSpectra = []
NearSpectra = []
FarHist = np.zeros([NumberOfVisibleEnergyBins])
OldFarHist = np.zeros([NumberOfVisibleEnergyBins])
NearHist = np.zeros([NumberOfVisibleEnergyBins])

if os.path.isfile("Bugey.npz"):
    print "Loading Bugey Values."
    Bugey4Samples = np.load("Bugey.npy")
else:
    Bugey4Samples = np.random.normal(5.752e-43, 8.0528e-45, NumberOfSamples)
    np.save("Bugey", Bugey4Samples)
    
NearAccHistogram = np.zeros([len(VisibleEnergyBinEdges)])
NearLi9Histogram = np.zeros([len(VisibleEnergyBinEdges)])
FarAccHistogram = np.zeros([len(VisibleEnergyBinEdges)])
FarLi9Histogram = np.zeros([len(VisibleEnergyBinEdges)])
OldFarAccHistogram = np.zeros([len(VisibleEnergyBinEdges)])
OldFarLi9Histogram = np.zeros([len(VisibleEnergyBinEdges)])

print ("Turn on pull terms for near background")

def LogLikelihood(NearAcc, NearNorm_A, NearNorm_B, NearNorm_C, Li9Rate, FNSMRate, NearLi9Rate, NearFNSMRate,
    Near_A, Near_B, Near_C, Far_A, Far_B, Far_C, Old_Far_A, Old_Far_B, Old_Far_C, BumpNorm, SSTT13, M13):
    global Count, OldBumpNorm, OldNearNorm_A, OldNearNorm_B, OldSSTT13, OldM13, OldLi9Rate, OldFNSMRate, OldNearLi9Rate
    global OldNearFNSMRate, OldBugey4Coeff, OldNearNorm_C    
    global R1Osc_, R2Osc_, R1NoOsc_, R1NoOsc_, NearR1Osc_, NearR2Osc_, NearR1NoOsc_, NearR1NoOsc_
    global OldFarSpectra, FarSpectra, NearSpectra, NearNormalization
    global FarHist, NearHist, OldFarHist
    global OldNear_A, OldNear_B, OldNear_C
    global OldFar_A, OldFar_B, OldFar_C
    global OldOldFar_A, OldOldFar_B, OldOldFar_C
    global NearAccHistogram, NearLi9Histogram, FarAccHistogram, FarLi9Histogram, OldFarAccHistogram, OldFarLi9Histogram

    NearNorm_B = np.fabs(NearNorm_B)
    NearNorm_A = np.fabs(NearNorm_A)
    NearNorm_C = np.fabs(NearNorm_C)

    start = time.time()   
    
    LikelihoodValue = 0.
    NoOscLikelihoodValue = 0.
    
    NearAcc = np.fabs(NearAcc)
    Li9Rate = np.fabs(Li9Rate)
    FNSMRate = np.fabs(FNSMRate)
    NearLi9Rate = np.fabs(NearLi9Rate)
    NearFNSMRate = np.fabs(NearFNSMRate)
    
    
    NoOscLikelihoodValue += .5*np.power((FNSMRate - FNSMRateMean)/FNSMRateSigma, 2.)
    NoOscLikelihoodValue += .5*np.power((NearFNSMRate - NearFNSMRateMean)/NearFNSMRateSigma, 2.)
    NoOscLikelihoodValue += .5*np.power((NearLi9Rate - NearLi9RateMean)/NearLi9Sigma, 2.)
    #NoOscLikelihoodValue += .5*np.power((Bugey4 - 5.752e-43)/8.0528e-45, 2)
       
    LikelihoodValue += .5*np.power((FNSMRate - FNSMRateMean)/FNSMRateSigma, 2.)
    if NearDet == True:
        if NearFNSMRate <= NearFNSMRateMean:
            LikelihoodValue += .5*np.power((NearFNSMRate - NearFNSMRateMean)/NearFNSMRateSigma, 2.)
        if NearLi9Rate <= NearLi9RateMean:
            LikelihoodValue += .5*np.power((NearLi9Rate - NearLi9RateMean)/NearLi9Sigma, 2.)
        #LikelihoodValue += .5*np.power((Bugey4 - 5.752e-43)/8.0528e-45, 2)
        if NearAcc <= NearAccMean:
            LikelihoodValue += .5*np.power((NearAcc - NearAccMean)/NearAccSigma, 2.)
    
    if (M13 > M13Mean):
        LikelihoodValue += .5*np.power((M13 - M13Mean)/M13RightSigma, 2.)
    else:
        LikelihoodValue += .5*np.power((M13 - M13Mean)/M13LeftSigma, 2.)
    
       
    LikelihoodValue += .5*np.power((Li9Rate - Li9RateMean)/Li9RateSigma, 2.)
    NoOscLikelihoodValue += .5*np.power((Li9Rate - Li9RateMean)/Li9RateSigma, 2.)
    
    if NearDet == True:
        N = np.array([Near_A, Near_B, Near_C]) - np.array([NearMean[0], NearMean[1], NearMean[2]])
        LikelihoodValue += .5*np.dot(N, np.dot(InvNearCov, N))
        NoOscLikelihoodValue += .5*np.dot(N, np.dot(InvNearCov, N))
    
    N = np.array([Far_A, Far_B, Far_C]) - np.array([FarMean[0], FarMean[1], FarMean[2]]) 
    LikelihoodValue += .5*np.dot(N, np.dot(InvFarCov, N))
    NoOscLikelihoodValue += .5*np.dot(N, np.dot(InvFarCov, N)) 
     
    N = np.array([Old_Far_A, Old_Far_B, Old_Far_C]) - np.array([OldFarMean[0], OldFarMean[1], OldFarMean[2]])
    LikelihoodValue += .5*np.dot(N, np.dot(InvOldFarCov, N))
    NoOscLikelihoodValue += .5*np.dot(N, np.dot(InvOldFarCov, N))
      
    if Near_A != OldNear_A or Near_B != OldNear_B or Near_C != OldNear_C or Count==0:
        NearHist = FitterFunctions.GetHistogram(NearData, Near_A, Near_B, Near_C, VisibleEnergyBinEdges)
        NearAccHistogram = FitterFunctions.GetHistogram(NearAccidentals, Near_A, Near_B, Near_C, VisibleEnergyBinEdges)
        NearAccHistogram = NearAccHistogram/np.float(np.sum(NearAccHistogram))
        NearLi9Histogram = FitterFunctions.GetHistogram(Li9Events, Near_A, Near_B, Near_C, VisibleEnergyBinEdges)
        NearLi9Histogram = NearLi9Histogram/np.float(np.sum(NearLi9Histogram))
        
        NearAccHistogram[np.isfinite(NearAccHistogram) == False] = 0.  
        NearLi9Histogram[np.isfinite(NearLi9Histogram) == False] = 0.  
        
        OldNear_A = Near_A
        OldNear_B = Near_B
        OldNear_C = Near_C
    if Far_A != OldFar_A or Far_B != OldFar_B or Far_C != OldFar_C or Count==0:
        FarHist = FitterFunctions.GetHistogram(FarData, Far_A, Far_B, Far_C, VisibleEnergyBinEdges)
        FarAccHistogram = FitterFunctions.GetHistogram(FarAccidentals, Far_A, Far_B, Far_C, VisibleEnergyBinEdges)
        FarAccHistogram = FarAccHistogram/np.float(np.sum(FarAccHistogram))
        FarLi9Histogram = FitterFunctions.GetHistogram(Li9Events, Far_A, Far_B, Far_C, VisibleEnergyBinEdges)
        FarLi9Histogram = FarLi9Histogram/np.float(np.sum(FarLi9Histogram))
        
        FarAccHistogram[np.isfinite(FarAccHistogram) == False] = 0.  
        FarLi9Histogram[np.isfinite(FarLi9Histogram) == False] = 0.
        
        
        OldFar_A = Far_A
        OldFar_B = Far_B
        OldFar_C = Far_C
    if OldFar_A != OldOldFar_A or OldFar_B != OldOldFar_B or OldFar_C != OldOldFar_C or Count==0:
        OldFarHist = FitterFunctions.GetHistogram(OldFarData, OldFar_A, OldFar_B, OldFar_C, VisibleEnergyBinEdges)
        OldFarAccHistogram = FitterFunctions.GetHistogram(OldFarAccidentals, OldFar_A, OldFar_B, OldFar_C, VisibleEnergyBinEdges)
        OldFarAccHistogram = OldFarAccHistogram/np.float(np.sum(OldFarAccHistogram))
        OldFarLi9Histogram = FitterFunctions.GetHistogram(Li9Events, OldFar_A, OldFar_B, OldFar_C, VisibleEnergyBinEdges)
        OldFarLi9Histogram = OldFarLi9Histogram/np.float(np.sum(OldFarLi9Histogram))
        
        OldFarAccHistogram[np.isfinite(OldFarAccHistogram) == False] = 0.  
        OldFarLi9Histogram[np.isfinite(OldFarLi9Histogram) == False] = 0.
        
        OldOldFar_A = OldFar_A
        OldOldFar_B = OldFar_B
        OldOldFar_C = OldFar_C
    
    ChiSquareCorrection = 0.
    

    
    l = len(NeutrinoEnergyBinEdges)-1

    BumpSpectrum_ = np.fabs(float(BumpNorm))*BumpSpectrum
    
    if OldNearNorm_A != NearNorm_A or OldNearNorm_B != NearNorm_B or OldNearNorm_C != NearNorm_C or Count==0:
        Size = RunFractions.size
        
        #NearNormalization = (1.+float(NearNorm_A))*np.ones([Size]) + float(NearNorm_B)*np.exp(float(NearNorm_C)*RunFractions)
        NearNormalization = FitterFunctions.NormFunction(Size, RunFractions, RunDelta, NearNorm_A, NearNorm_B, NearNorm_C)
        if np.any(NearNormalization != NearNormalization):
            return np.inf
        if np.any(NearNormalization < 0.):
            return np.inf
        OldNearNorm_A = NearNorm_A
        OldNearNorm_B = NearNorm_B
        OldNearNorm_C = NearNorm_C
    
    if (SSTT13 != OldSSTT13) or (M13 != OldM13) or (BumpNorm != OldBumpNorm) or Count==0:
        pool = mp.Pool(processes=4)
        R1Args = (0, SSTT13, M13, R1FarCenters, R2FarCenters, R1FarHistogram, R2FarHistogram)
        R2Args = (1, SSTT13, M13, R1FarCenters, R2FarCenters, R1FarHistogram, R2FarHistogram)
        R1 = (0, R1FarCenters, R2FarCenters, R1FarHistogram, R2FarHistogram)
        R2 = (1, R1FarCenters, R2FarCenters, R1FarHistogram, R2FarHistogram)
        R1Osc_ = np.asarray([pool.apply(Spectrum, args=(i, R1Args)) for i in range(l)])
        R2Osc_ = np.asarray([pool.apply(Spectrum, args=(i, R2Args)) for i in range(l)])
        R1NoOsc_ = np.asarray([pool.apply(NoOscSpectrum, args=(i, R1)) for i in range(l)])
        R2NoOsc_ = np.asarray([pool.apply(NoOscSpectrum, args=(i, R2)) for i in range(l)])

        NearR1Args = (0, SSTT13, M13, R1NearCenters, R2NearCenters, R1NearHistogram, R2NearHistogram)
        NearR2Args = (1, SSTT13, M13, R1NearCenters, R2NearCenters, R1NearHistogram, R2NearHistogram)
        NearR1 = (0, R1NearCenters, R2NearCenters, R1NearHistogram, R2NearHistogram)
        NearR2 = (1, R1NearCenters, R2NearCenters, R1NearHistogram, R2NearHistogram)
        NearR1Osc_ = np.asarray([pool.apply(Spectrum, args=(i, NearR1Args)) for i in range(l)])
        NearR2Osc_ = np.asarray([pool.apply(Spectrum, args=(i, NearR2Args)) for i in range(l)])
        NearR1NoOsc_ = np.asarray([pool.apply(NoOscSpectrum, args=(i, NearR1)) for i in range(l)])
        NearR2NoOsc_ = np.asarray([pool.apply(NoOscSpectrum, args=(i, NearR2)) for i in range(l)])
        pool.close()
        pool.join()
        pool.terminate()
         
        
        NumberOfTrueBins = len(NeutrinoEnergyBinCenters)
        OldFarR1IsotopeOsc_ = FitterFunctions.CorrectSpectrum(OldFarCorrections, R1Osc_, NumberOfTrueBins, len(OldFarEventList), 0)
        OldFarR2IsotopeOsc_ = FitterFunctions.CorrectSpectrum(OldFarCorrections, R2Osc_, NumberOfTrueBins, len(OldFarEventList), 1)
        OldFarR1IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(OldFarCorrections, R1NoOsc_, NumberOfTrueBins, len(OldFarEventList), 0)
        OldFarR2IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(OldFarCorrections, R2NoOsc_, NumberOfTrueBins, len(OldFarEventList), 1)

        FarR1IsotopeOsc_ = FitterFunctions.CorrectSpectrum(FarCorrections, R1Osc_, NumberOfTrueBins, len(FarEventList), 0)
        FarR2IsotopeOsc_ = FitterFunctions.CorrectSpectrum(FarCorrections, R2Osc_, NumberOfTrueBins, len(FarEventList), 1)
        FarR1IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(FarCorrections, R1NoOsc_, NumberOfTrueBins, len(FarEventList), 0)
        FarR2IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(FarCorrections, R2NoOsc_, NumberOfTrueBins, len(FarEventList), 1)

        NumVetoed = 3
        NearR1IsotopeOsc_ = FitterFunctions.CorrectSpectrum(NearCorrections, NearR1Osc_, NumberOfTrueBins, len(NearEventList)+NumVetoed, 0)
        NearR2IsotopeOsc_ = FitterFunctions.CorrectSpectrum(NearCorrections, NearR2Osc_, NumberOfTrueBins, len(NearEventList)+NumVetoed, 1)
        NearR1IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(NearCorrections, NearR1NoOsc_, NumberOfTrueBins, len(NearEventList)+NumVetoed, 0)
        NearR2IsotopeNoOsc_ = FitterFunctions.CorrectSpectrum(NearCorrections, NearR2NoOsc_, NumberOfTrueBins, len(NearEventList)+NumVetoed, 1)
        
        R1OscNewFlux_ = np.multiply(BumpSpectrum_.T, R1Osc_).T
        R2OscNewFlux_ = np.multiply(BumpSpectrum_.T, R2Osc_).T
        NearR1OscNewFlux_ = np.multiply(BumpSpectrum_.T, NearR1Osc_).T
        NearR2OscNewFlux_ = np.multiply(BumpSpectrum_.T, NearR2Osc_).T

        pool = mp.Pool(processes=4)        
        def f(x):
            W = FitterFunctions.GetFullSpectrum(0, OldFarR1IsotopeOsc_, OldFarR2IsotopeOsc_, 
            OldFarR1IsotopeNoOsc_, OldFarR2IsotopeNoOsc_, Fluxes[0,:,x] , Fluxes[1,:,x], Fluxes[2,:,x], Fluxes[3,:,x], 
            Bugey4Samples[x], R1FarMean, R2FarMean, Bugey4Fractions, R1Fissions[x:x+1, :,:], R2Fissions[x:x+1, :,:], OldFarEfficiencies[x], 
            LivetimeRatio, FDI_Hs[x], R1OscNewFlux_[:,x], R2OscNewFlux_[:,x])
            P = np.sum(W[0], axis=1)
            
            U = FitterFunctions.GetFullSpectrum(0, FarR1IsotopeOsc_, FarR2IsotopeOsc_, 
            FarR1IsotopeNoOsc_, FarR2IsotopeNoOsc_, Fluxes[0,:,x] , Fluxes[1,:,x], Fluxes[2,:,x], Fluxes[3,:,x], 
            Bugey4Samples[x], R1FarMean, R2FarMean, Bugey4Fractions, NewR1Fissions[x:x+1, :,:], NewR2Fissions[x:x+1, :,:], FarEfficiencies[x], 
            NewFarLivetimeRatio, FDII_Hs[x], R1OscNewFlux_[:,x], R2OscNewFlux_[:,x])
            X = np.sum(U[0], axis=1)
            
            G = FitterFunctions.GetFullSpectrum(1, NearR1IsotopeOsc_, NearR2IsotopeOsc_, 
            NearR1IsotopeNoOsc_, NearR2IsotopeNoOsc_, Fluxes[0,:,x] , Fluxes[1,:,x], Fluxes[2,:,x], Fluxes[3,:,x], 
            Bugey4Samples[x], R1NearMean, R2NearMean, Bugey4Fractions, NearR1Fissions[x:x+1, :,:], NearR2Fissions[x:x+1, :,:], NearEfficiencies[x], 
            NewNearLivetimeRatio, ND_Hs[x], NearR1OscNewFlux_[:,x], NearR2OscNewFlux_[:,x])
            Q = np.multiply(NDVeto, G[0])
            
            return P, X, Q
            
        Output = [pool.apply_async(f, args=(x,)) for x in range(NumberOfSamples)]
        Res = [p.get() for p in Output]
        OldFarSpectra = []
        FarSpectra = []
        NearSpectra = []       
        for R in Res:
            OldFarSpectra.append(R[0])
            FarSpectra.append(R[1])
            NearSpectra.append(R[2])
        
        OldSSTT13 = SSTT13
        OldM13 = M13
        OldBumpNorm = BumpNorm
        pool.close()
        pool.join()
        pool.terminate()
        
    pool = mp.Pool(processes=4) 
    def GetTotalSpectra(OldSpectra, Spectra, NearSpectra, OldBackground, Background, NearBackground):
        A = OldSpectra + OldBackground
        B = Spectra + Background
        C = NearSpectra + NearBackground
        return A, B, C

    print "Combining Spectra"
    Results = [pool.apply_async(GetTotalSpectra, args=(OldFarSpectra[x], FarSpectra[x], np.sum(np.multiply(NearSpectra[x], NearNormalization), axis=1),
    np.sum(np.outer(0.069*OldFarAccHistogram + Li9Rate*OldFarLi9Histogram + FNSMRate*FNHistograms[x], LivetimeRatio*OldFarRuntimes*1.15741e-5), axis=1),
    np.sum(np.outer(0.118*FarAccHistogram + Li9Rate*FarLi9Histogram + FNSMRate*FNHistograms[x], NewFarLivetimeRatio*FarRuntimes*1.15741e-5), axis=1),
    np.sum(np.multiply(np.outer(NearAcc*NearAccHistogram + NearLi9Rate*NearLi9Histogram + NearFNSMRate*FNHistograms[x], NewNearLivetimeRatio*NearRuntimes*1.15741e-5), np.multiply(NDVeto, NearNormalization)), axis=1)
    ) ) for x in range(NumberOfSamples)]

    Spect = [R.get() for R in Results]
    pool.close()
    pool.join()
    pool.terminate()

    def GetLikelihoods(OldSpectra, Spectra, NearSpectra, OldData, Data, NearData):
        #print AdvancedPoisson(OldSpectra, OldData).shape
        OldSpectra[OldSpectra != OldSpectra] = 0.
        Spectra[Spectra != Spectra] = 0.
        NearSpectra[NearSpectra != NearSpectra] = 0.
        
        OldSpectra[OldSpectra < 0.] = 0.
        Spectra[Spectra < 0.] = 0.
        NearSpectra[NearSpectra < 0.] = 0.
        
        S = np.sum(FitterFunctions.AdvancedPoisson(OldSpectra, OldData)) + np.sum(FitterFunctions.AdvancedPoisson(Spectra, Data)) + np.sum(FitterFunctions.AdvancedPoisson(NearSpectra, NearData)) #FitterFunctions.GetProbability1(NearSpectra, NearData)
        return S
        
    print "Getting Likelihoods"
    pool = mp.Pool(processes=4) 
    Results = [pool.apply_async(GetLikelihoods, args=( (Spect[x])[0], (Spect[x])[1], (Spect[x])[2], OldFarHist, FarHist, NearHist)) for x in range(NumberOfSamples)]
    LogLikelihoods = [R.get() for R in Results]
    pool.close()
    pool.join()
    pool.terminate()
    
    
    def GetChiSquares(OldSpectra, Spectra, NearSpectra, OldData, Data, NearData):
        OldSpectra[OldSpectra != OldSpectra] = 0.
        Spectra[Spectra != Spectra] = 0.
        NearSpectra[NearSpectra != NearSpectra] = 0.
        
        OldSpectra[OldSpectra < 0.] = 0.
        Spectra[Spectra < 0.] = 0.
        NearSpectra[NearSpectra < 0.] = 0.

        S = np.sum(FitterFunctions.Chi(OldSpectra, OldData)) + np.sum(FitterFunctions.Chi(Spectra, Data)) + np.sum(FitterFunctions.Chi(NearSpectra, NearData))
        return S
       
    ChiS = 0. 
    pool = mp.Pool(processes=4) 
    Results = [pool.apply_async(GetChiSquares, args=( (Spect[x])[0], (Spect[x])[1], (Spect[x])[2], OldFarHist, FarHist, NearHist)) for x in range(NumberOfSamples)]
    Chis = [RR.get() for RR in Results]
    pool.close()
    pool.join()
    pool.terminate()

    ChiArray = np.asarray(Chis)
    np.save("Plots/Chi/" + str(Count), ChiArray)

    for out in Chis:
        ChiS += out
    ChiS /= NumberOfSamples

    
    HG = LikelihoodValue
    
      
    Sum = mpmath.mpf(0.)
    NonOscSum = mpmath.mpf(0.)
    TotalNewFar = 0.
    TotalNewNear = 0.
    TotalFarFound = 0
    TotalNearFound = 0

    TotalThirdPub = 0.
    TotalThirdPubNonOsc = 0.
    TotalThirdPubFound = 0
    TotalSpectrum = np.zeros([NumberOfVisibleEnergyBins])
    Samples = []

    OldSpectraSamples = []
    FarSpectraSamples = []
    NearSpectraSamples = []

    for out in LogLikelihoods:
        Val = mpmath.mpf(-out)
        Exp = mpmath.exp(Val)
        Sum = mpmath.fadd(Sum, Exp)
        Samples.append(-2.*out)     
        
    #print Samples
    plt.hist(Samples, bins=100)
    plt.savefig("Plots/Likelihood/" + str(Count) + ".png")
    plt.clf()
    
    plt.hist(Chis, bins=100)
    plt.savefig("Plots/Likelihood/Chi" + str(Count) + ".png")
    plt.clf()    
    
    
    OldSpectras = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    OldRatio = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    for i in range(NumberOfSamples):
        S = (Spect[i])[0]
        for j in range(NumberOfVisibleEnergyBins):
            OldSpectras[i][j] = (S)[j]
            OldRatio[i][j] = (OldFarHist[j]-OldSpectras[i][j])/OldSpectras[i][j]
    
    plt.errorbar(VisibleEnergyBinCenters, np.mean(OldSpectras, axis=0), yerr=np.std(OldSpectras, axis=0), linestyle='None', marker='.', label="Theory")
    plt.errorbar(VisibleEnergyBinCenters, OldFarHist, yerr=np.sqrt(OldFarHist), linestyle='None', marker='.', label="Experiment")
    plt.ylabel("Number Of Events")
    plt.xlabel("Visible Energy (MeV)")
    plt.legend()
    plt.title("DCThirdPub Data")
    plt.savefig("Plots/Spectra/OldFar_" +str(Count) + ".png")
    plt.clf()
    plt.errorbar(VisibleEnergyBinCenters, np.mean(OldRatio, axis=0), yerr=np.std(OldRatio, axis=0), linestyle='None', marker='.')
    plt.ylabel("Excess")
    plt.xlabel("Visible Energy (MeV)") 
    plt.title("DCThirdPub Data Excess Relative to Theory")
    plt.savefig("Plots/Spectra/OldFarExcess_" +str(Count) + ".png")
    plt.clf()   
    

    FarSpectras = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    Ratio = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    for i in range(NumberOfSamples):
        S = (Spect[i])[1]
        for j in range(NumberOfVisibleEnergyBins):
            FarSpectras[i][j] = (S)[j]
            Ratio[i][j] = (FarHist[j]-FarSpectras[i][j])/FarSpectras[i][j]

    plt.errorbar(VisibleEnergyBinCenters, np.mean(FarSpectras, axis=0), yerr=np.std(FarSpectras, axis=0), linestyle='None', marker='.', label="Theory")
    plt.errorbar(VisibleEnergyBinCenters, FarHist, yerr=np.sqrt(FarHist), linestyle='None', marker='.', label="Experiment")
    plt.ylabel("Number Of Events")
    plt.xlabel("Visible Energy (MeV)")
    plt.title("9 Month Far Data")
    plt.legend()
    plt.savefig("Plots/Spectra/Far_" + str(Count) + ".png")
    plt.clf()
    
    plt.errorbar(VisibleEnergyBinCenters, np.mean(Ratio, axis=0), yerr=np.std(Ratio, axis=0), linestyle='None', marker='.')
    plt.ylabel("Excess")
    plt.xlabel("Visible Energy (MeV)") 
    plt.title("9 Month Far Data Excess Relative to Theory")
    plt.savefig("Plots/Spectra/FarExcess_" +str(Count) + ".png")
    plt.clf()
    
    NearSpectras = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    NearRatio = np.zeros([NumberOfSamples, NumberOfVisibleEnergyBins])
    for i in range(NumberOfSamples):
        S = (Spect[i])[2]
        for j in range(NumberOfVisibleEnergyBins):
            NearSpectras[i][j] = (S)[j]
            NearRatio[i][j] = (NearHist[j]-NearSpectras[i][j])/NearSpectras[i][j]

    plt.errorbar( VisibleEnergyBinCenters, np.mean(NearSpectras, axis=0), yerr=np.std(NearSpectras, axis=0), linestyle='None', marker='.', label="Theory")
    plt.errorbar(VisibleEnergyBinCenters, NearHist, yerr=np.sqrt(NearHist), linestyle='None', marker='.', label="Experiment")
    plt.ylabel("Number Of Events")
    plt.xlabel("Visible Energy (MeV)")
    plt.title("9 Month Near Data")
    plt.legend()
    plt.savefig("Plots/Spectra/Near_" + str(Count) + ".png")
    plt.clf()
    
    plt.errorbar(VisibleEnergyBinCenters, np.mean(NearRatio, axis=0), yerr=np.std(NearRatio, axis=0), linestyle='None', marker='.')
    plt.ylabel("Excess")
    plt.xlabel("Visible Energy (MeV)") 
    plt.title("9 Month Near Data Excess Relative to Theory")
    plt.savefig("Plots/Spectra/NearExcess_" +str(Count) + ".png")
    plt.clf()
    
    Sum = mpmath.fdiv(Sum, NumberOfSamples)    
    LikelihoodValue += -2.*mpmath.log(Sum) 

    #Like = 2.*LikelihoodValue
    #LikelihoodValue = Like 

    TotalSpectrum /= NumberOfSamples

    TotalNewFar = np.sum(np.mean(FarSpectras, axis=0))
    TotalNewNear = np.sum(np.mean(NearSpectras, axis=0))
    TotalFarFound /= NumberOfSamples
    TotalNearFound /= NumberOfSamples

    TotalThirdPub = np.sum(np.mean(OldSpectras, axis=0))
    TotalThirdPubNonOsc /= NumberOfSamples
    TotalThirdPubFound /= NumberOfSamples
    
    
    
    if LikelihoodValue != LikelihoodValue:
        LikelihoodValue = 1.e40
    
    end = time.time()
    Count += 1
    del pool
    print "SSTT13, M13", (SSTT13, M13)
    print "Far Li9 Rate and FNSM Rate: ", Li9Rate, FNSMRate 
    print "Near Li9 Rate and FNSM Rate: ", NearLi9Rate, NearFNSMRate
    print "Near Acc Rate: ", NearAcc
    print "Bump Norm: ", BumpNorm
    print "Near Norm Co-efficients: ", NearNorm_A, NearNorm_B, NearNorm_C
    print "Near Energy Scale: ", Near_A, Near_B, Near_C
    print "Old Far Energy Scale", Old_Far_A, Old_Far_B, Old_Far_C
    print "Far Energy Scale: ", Far_A, Far_B, Far_C
    print "Number Of Iterations so far: ", Count
    print "New Far Data (Prediction, Found, Ratio): ", TotalNewFar, np.sum(FarHist), TotalNewFar/np.sum(FarHist)
    print "New Near Data (Prediction, Found, Ratio): ", TotalNewNear, np.sum(NearHist), TotalNewNear/np.sum(NearHist)
    print "Third Pub Far Data (Prediction, Found, Ratio): ", TotalThirdPub, np.sum(OldFarHist), TotalThirdPub/np.sum(OldFarHist)
    print " "
    NDOF = 3*NumberOfVisibleEnergyBins - 1 +13
    print "ChiSquare Correction: ", ChiSquareCorrection
    print "Chi, Degrees Of Freedom, Chi Per Degree: ", ChiS, NDOF, ChiS/NDOF, HG
    print "Likelihood: ", LikelihoodValue
    print "Minimization pass took: ", (end-start)/60., " minutes"
    print ' '
    return LikelihoodValue
    
def SimplexFunc(x):
    NearAcc, NearNorm_A, NearNorm_B, NearNorm_C, Li9Rate, FNSMRate, NearLi9Rate, NearFNSMRate, Near_A, Near_B, Near_C, Far_A, Far_B, Far_C, Old_Far_A, Old_Far_B, Old_Far_C, BumpNorm, SSTT13, M13 = x
    Val = LogLikelihood(NearAcc, NearNorm_A, NearNorm_B, NearNorm_C, Li9Rate, FNSMRate, NearLi9Rate, NearFNSMRate, Near_A, Near_B, Near_C, Far_A, Far_B, Far_C, Old_Far_A, Old_Far_B, Old_Far_C, BumpNorm, SSTT13, M13)
    return Val
    


def main():
    global Count
    
    
    x0 = np.array([0.301670059774, 0.994252132593, 0.00230538402151, 0.00590002382274, 0.8945971901, 0.464840905602, 8.25381559741, 3.94217843788,
       NearMean[0], NearMean[1], NearMean[2], FarMean[0], FarMean[1], FarMean[2], OldFarMean[0], OldFarMean[1], OldFarMean[2], 0.00453978243062, 0.10091505595897542, 0.0024079639515751355])
    print "Make sure you have the proton normalization for the fiducial cut and the spill in/spill out normalization correctly."
    print "Minimizing"
    
    #res = minimize(SimplexFunc, x0, method='Powell', options={'xtol': 1e-8, 'disp': True})
    Count = 0
    
    #LOOK AT NEAR CUT
    print "MAKE SURE NEARDET IS SET TO TRUE IF NEAR DETECTOR ANALYSIS and UNCOMMENT OUT LIKELIHOODS"

    
    m = Minuit(LogLikelihood,
        SSTT13= 0.12054415908568761, M13= 0.0025043194579322248, Li9Rate = 1.1244941107796325, FNSMRate = 0.43741890871782574, NearLi9Rate = 7.901068678691314, NearFNSMRate = 3.7788535947724244, 
        error_SSTT13 = 5., limit_SSTT13=(0., 1.), limit_M13=(0., 1.), limit_Li9Rate=(0., 10.), limit_FNSMRate=(0., 1.), limit_NearLi9Rate=(0., 10.), 
        limit_NearFNSMRate=(0., 10.), 
        error_Li9Rate = 4., error_FNSMRate = 4., 
        error_M13 = 2.e-2, error_NearLi9Rate = 4., error_NearFNSMRate = 4., errordef = 1.,
        #fix_NearLi9Rate=True, fix_NearFNSMRate = True,
        BumpNorm = 0.011238038728963017, error_BumpNorm = 1., fix_BumpNorm=True,
        NearNorm_A =3.969585771201878e-11, NearNorm_B = 0.0, NearNorm_C = 0.0, 
        limit_NearNorm_A=(0., 1.), limit_NearNorm_B=(0., 1.),
        #limit_NearNorm_A=(0., 10.), limit_NearNorm_B=(0., 10.), limit_NearNorm_C=(0., 10.),
        NearAcc = 0.575028741826883, error_NearAcc = 1.5,
        #fix_NearAcc=True,
        error_NearNorm_A=4., error_NearNorm_B=4., error_NearNorm_C=5.,
        #fix_NearNorm_A=True,
        fix_NearNorm_B=True, fix_NearNorm_C=True,
        #fix_NearLi9Rate=True, fix_NearFNSMRate=True, 
        
        Near_A = -0.004116474803739965, Near_B = 1.0018032274975033, Near_C = -1.7993784835962714e-07, error_Near_A = 3., error_Near_B=.1, error_Near_C=1.e-5,
        Far_A = -0.012606607718833054, Far_B =  1.0056769330102382, Far_C = -3.662679490821011e-08, error_Far_A = 1., error_Far_B=.1, error_Far_C=1.e-5,
        Old_Far_A = -0.012601515890730806, Old_Far_B = 1.0056896190670936, Old_Far_C = -1.0659506576591107e-07, error_Old_Far_A = 1., error_Old_Far_B=.1,
        
        #Near_A = NearMean[0], Near_B = NearMean[1], Near_C = NearMean[2], error_Near_A = 3., error_Near_B=.1, error_Near_C=1.e-5,
        #Far_A = FarMean[0], Far_B = FarMean[1], Far_C = FarMean[2], error_Far_A = 1., error_Far_B=.1, error_Far_C=1.e-5,
        #Old_Far_A = OldFarMean[0], Old_Far_B = OldFarMean[1], Old_Far_C = OldFarMean[2], error_Old_Far_A = 1., error_Old_Far_B=.1, error_Old_Far_C=1.e-5,
        #fix_Near_A=True, fix_Near_B=True, fix_Near_C=True,
        #fix_Far_A=True, fix_Far_B=True, fix_Far_C=True, fix_Old_Far_A=True, fix_Old_Far_B=True, fix_Old_Far_C=True
        #fix_SSTT13=True, fix_M13=True, fix_Li9Rate=True, fix_FNSMRate=True, fix_NearLi9Rate=True,
        #fix_Near_A=True, fix_Near_B=True, fix_Near_C=True, 
        #fix_Far_A=True, fix_Far_B=True, fix_Far_C=True, fix_Old_Far_A=True,
        #fix_Old_Far_B=True, fix_Old_Far_C=True,
        limit_Near_A=(NearMean[0]-5.*np.sqrt(NearCov[0][0]), NearMean[0]+5.*np.sqrt(NearCov[0][0])), 
        limit_Near_B=(NearMean[1]-5.*np.sqrt(NearCov[1][1]), NearMean[1]+5.*np.sqrt(NearCov[1][1])),
        limit_Near_C=(NearMean[2]-5.*np.sqrt(NearCov[2][2]), NearMean[2]+5.*np.sqrt(NearCov[2][2])),
        
        limit_Far_A=(FarMean[0]-5.*np.sqrt(FarCov[0][0]), FarMean[0]+5.*np.sqrt(FarCov[0][0])),
        limit_Far_B=(FarMean[1]-5.*np.sqrt(FarCov[1][1]), FarMean[1]+5.*np.sqrt(FarCov[1][1])),
        limit_Far_C=(FarMean[2]-5.*np.sqrt(FarCov[2][2]), FarMean[2]+5.*np.sqrt(FarCov[2][2])),
        
        limit_Old_Far_A=(OldFarMean[0]-5.*np.sqrt(OldFarCov[0][0]), OldFarMean[0]+5.*np.sqrt(OldFarCov[0][0])),
        limit_Old_Far_B=(OldFarMean[1]-5.*np.sqrt(OldFarCov[1][1]), OldFarMean[1]+5.*np.sqrt(OldFarCov[1][1])),
        limit_Old_Far_C=(OldFarMean[2]-5.*np.sqrt(OldFarCov[2][2]), OldFarMean[2]+5.*np.sqrt(OldFarCov[2][2]))
        )
        
    m.migrad()
    m.print_param()
    print('value', m.values)
    print('covariance', m.covariance)
    #m.profile('SSTT13')
    
    mi = .07
    ma = .14
    d = .001
    N = int((ma-mi)/d)
    '''
    for i in range(N+1):
        SS = mi + i*d
        LogLikelihood(m.values['NearAcc'], m.values['NearNorm_A'], m.values['NearNorm_B'], m.values['NearNorm_C'], m.values['Li9Rate'], m.values['FNSMRate'], m.values['NearLi9Rate'], m.values['NearFNSMRate'],
        m.values['Near_A'], m.values['Near_B'], m.values['Near_C'], m.values['Far_A'], m.values['Far_B'], m.values['Far_C'], m.values['Old_Far_A'], m.values['Old_Far_B'], m.values['Old_Far_C'], m.values['BumpNorm'], 
        SS, m.values['M13'])
    
    
        LogLikelihood(m.values['NearAcc'], m.values['NearNorm_A'], m.values['NearNorm_B'], m.values['NearNorm_C'], m.values['Li9Rate'], m.values['FNSMRate'], m.values['NearLi9Rate'], m.values['NearFNSMRate'],
    m.values['Near_A'], m.values['Near_B'], m.values['Near_C'], m.values['Far_A'], m.values['Far_B'], m.values['Far_C'], m.values['Old_Far_A'], m.values['Old_Far_B'], m.values['Old_Far_C'], m.values['BumpNorm'], 
    SS, m.values['M13'])
    '''
main() 
''''
        NearA = NearMean[0], NearB=NearMean[1], NearC=NearMean[2], error_NearA=2., error_NearB=.1, error_NearC=1.e-3,
        FarA = FarMean[0], FarB=FarMean[1], FarC=FarMean[2], error_FarA=2., error_FarB=.1, error_FarC=1.e-3,
        OldFarA =OldFarMean[0], OldFarB=OldFarMean[1], OldFarC=OldFarMean[2], error_OldFarA=2., error_OldFarB=.01, error_OldFarC=1.e-3,
'''
