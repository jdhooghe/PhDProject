import numpy as np
from numba import jit
import scipy
from math import factorial, log
from scipy.integrate import quad
import os.path

K = .961e-43
Mn = 939.565
Mp = 938.272
Me = .510998
Delta = Mn - Mp
 
@jit(nopython=True)
def GetSampleProbability(OldSpectra, Spectra, NearSpectra, OldFarHistogram, FarHistogram, NearHistogram):
    A = np.sum(FitterFunctions.AdvancedPoisson(OldSpectra, OldFarHistogram)) + np.sum(FitterFunctions.AdvancedPoisson(Spectra, FarHistogram)) + np.sum(FitterFunctions.AdvancedPoisson(NearSpectra, NearHistogram))
    return A    

@jit
def FindEnergyCorrections(E, C, B, A):
    E = np.asarray(E)
    Shift = E - C*np.ones([E.size])
    
    C0 = Shift/B
    C1 = - np.power(Shift, 2)/(B*B*B)
    C2 = 2.*np.power(Shift, 3)/(B*B*B*B*B)
    
    Val = C0 + C1*A + C2*A*A
    
    #Val = C*np.ones([E.size]) + B*E + A*np.power(E, 2)
    return Val

#Unable to be jit'ed
def GetHistogram(Data, A, B, C, VisibleEnergyBinEdges): 
    Events = FindEnergyCorrections(Data, A, B, C)
    return np.histogram(Events, bins=VisibleEnergyBinEdges)[0]
    
def SterlingDelta(Found):
    Delta = Delta = .5*np.log(2*np.pi*Found) + (1./12.)*np.power(Found, -1.) - (1./360.)*np.power(Found, -3.)
    
def AdvancedPoisson(Expected, Found):
    
    #Mult = np.divide(np.power(Expected-Found, 2.), Expected)
    Expected[Expected < 0.] = 0.  
    
    N = [log(factorial(l)) for l in Found]
    N = np.asarray(N)   

    with np.errstate(invalid='ignore'):
        with np.errstate(divide='ignore'):
            Mult1 = np.log(Expected)
            Mult1[np.isfinite(Mult1) == False] = -1.0e30 
            Mult1 = -Mult1*Found + Expected
            Mult = Mult1 + N

    return Mult
    
def Chi(Expected, Found):
    
    NewExp = []
    NewData = []
    
    for i in range(28):
        NewExp.append(Expected[i])
        NewData.append(Found[i])
    
    Count = 0
    for i in range(6):
        Bin = 28 + Count
        #print Bin
        NewData.append(Found[Bin]+Found[Bin+1]+Found[Bin+2]+Found[Bin+3]+Found[Bin+4]+Found[Bin+6])
        NewExp.append(Expected[Bin]+Expected[Bin+1]+Expected[Bin+2]+Expected[Bin+3]+Expected[Bin+4]+Expected[Bin+6])
        Count+=6
    Expected_ = np.asarray(NewExp)
    Found_ = np.asarray(NewData)
    
    Diff = Expected_ - Found_
    Diff = np.power(Diff, 2.)
    with np.errstate(invalid='ignore'):
        Diff /= Expected_
        Diff[np.isfinite(Diff) == False] = 0. 
    return Diff
    
@jit(nopython=True)
def AdvancedPoisson1(Expected, Found):
    
    #Mult = np.divide(np.power(Expected-Found, 2.), Expected)
    Expected[Expected < 0.] = 0.
    Ratio = Found/Expected
    Mult = Found*np.log(Ratio)
    Mult[np.isfinite(Mult) == False] = 1. 
    
    return Mult    

#Unable to be jit'ed
def NormFunction(Size, x, Delta, A, B, C):
    G = (1.+A)*np.ones([Size]) + B*x + C*np.power(x, 2)
    return G
    #return np.multiply(G, Delta)*(1.e4)
    #return (1.+A)*np.ones([Size]) + B*np.exp(C*x)

@jit(nopython=True)
def GetProbability(N, Spectrum, Hist, dE):
    Prob = - np.sum(Spectrum) - np.log(factorial(N))
    Prob += np.dot(Hist, np.log(Spectrum))
    return Prob

def GetProbability1(Spectrum, Hist):
    Sum = np.sum(Spectrum)
    with np.errstate(divide='ignore'):
        Log = np.log(Spectrum/Sum)
        Log[Log == np.inf] = 0.
        Log[Log == -np.inf] = 0.
    Prob = np.dot(Hist, Log)
    return Prob

#Unable to be jit'ed    
def GetNumberOfEventsForEachRun(EventList):
    Numbers = np.zeros([len(EventList)])
    for i in range(len(EventList)):
        Numbers[i] = len(EventList[i])
    return Numbers
    

def CorrectSpectrum(Corrections, Base, NumberOfBins, NumberOfRuns, Reactor):
    Isotope = np.zeros([4, NumberOfRuns, NumberOfBins])
    Isotope[0, :] = np.multiply(Corrections[Reactor,0,:,:], Base)
    Isotope[1, :] = np.multiply(Corrections[Reactor,1,:,:], Base)
    Isotope[2, :] = np.multiply(Corrections[Reactor,2,:,:], Base)
    Isotope[3, :] = np.multiply(Corrections[Reactor,3,:,:], Base)
    return Isotope
    
def GetFinalSpectrum(Det, Bugey4, Bugey4Fractions, R1Fiss, R2Fiss, R1IsoOsc, R2IsoOsc, R1IsoNoOsc, R2IsoNoOsc, R1Mean, R2Mean, Efficiency, Livetime, R1BumpSpectrum, R2BumpSpectrum):
    
    if Det == 0:
        Prot = 6.739e29
    else:
        Frac = 1. #0.6131167
        Prot = Frac*6.767e29
    
    R1Fiss = R1Fiss[0, :, :]
    R2Fiss = R2Fiss[0, :, :]
    R1Fiss_ = np.transpose(R1Fiss) #(NumberOfRuns, Isotope)
    R2Fiss_ = np.transpose(R2Fiss)

    
    R1FissionTotals = np.asarray(np.sum(R1Fiss, axis=0)) #(NumberOfRuns, )
    R2FissionTotals = np.asarray(np.sum(R2Fiss, axis=0)) 
    
    R1Tot = np.sum(R1FissionTotals)
    R2Tot = np.sum(R2FissionTotals)
    
    Size = R1Fiss[0,:].size 
    
    R1_a_U235 = np.multiply(R1Fiss[0,:] - Bugey4Fractions[0]*R1FissionTotals, np.sum(R1IsoNoOsc[0,:,:], axis=1))
    R1_a_U238 = np.multiply(R1Fiss[1,:] - Bugey4Fractions[1]*R1FissionTotals, np.sum(R1IsoNoOsc[1,:,:], axis=1))
    R1_a_Pu239 = np.multiply(R1Fiss[2,:] - Bugey4Fractions[2]*R1FissionTotals, np.sum(R1IsoNoOsc[2,:,:], axis=1))
    R1_a_Pu241 = np.multiply(R1Fiss[3,:] - Bugey4Fractions[3]*R1FissionTotals, np.sum(R1IsoNoOsc[3,:,:], axis=1))
    
    R2_a_U235 = np.multiply(R2Fiss[0,:] - Bugey4Fractions[0]*R2FissionTotals, np.sum(R2IsoNoOsc[0,:,:], axis=1))
    R2_a_U238 = np.multiply(R2Fiss[1,:] - Bugey4Fractions[1]*R2FissionTotals, np.sum(R2IsoNoOsc[1,:,:], axis=1))
    R2_a_Pu239 = np.multiply(R2Fiss[2,:] - Bugey4Fractions[2]*R2FissionTotals, np.sum(R2IsoNoOsc[2,:,:], axis=1))
    R2_a_Pu241 = np.multiply(R2Fiss[3,:] - Bugey4Fractions[3]*R2FissionTotals, np.sum(R2IsoNoOsc[3,:,:], axis=1))
    
    R1Bugey = Prot*(Bugey4*InverseSquare(R1Mean)*R1FissionTotals + R1_a_U235 + R1_a_U238 + R1_a_Pu239 + R1_a_Pu241)  #(NumberOfRuns, )
    R2Bugey = Prot*(Bugey4*InverseSquare(R2Mean)*R2FissionTotals + R2_a_U235 + R2_a_U238 + R2_a_Pu239 + R2_a_Pu241)  #(NumberOfRuns, )      
    
    R1OscSpectrum = np.multiply(R1Fiss[0,:], R1IsoOsc[0,:,:].T) + np.multiply(R1Fiss[1,:], R1IsoOsc[1,:,:].T) + np.multiply(R1Fiss[2,:], R1IsoOsc[2,:,:].T) + np.multiply(R1Fiss[3,:], R1IsoOsc[3,:,:].T) 
    R2OscSpectrum = np.multiply(R2Fiss[0,:], R2IsoOsc[0,:,:].T) + np.multiply(R2Fiss[1,:], R2IsoOsc[1,:,:].T) + np.multiply(R2Fiss[2,:], R2IsoOsc[2,:,:].T) + np.multiply(R2Fiss[3,:], R2IsoOsc[3,:,:].T) 
    
    R1Spectrum = np.multiply(R1Fiss[0,:], R1IsoNoOsc[0,:,:].T) + np.multiply(R1Fiss[1,:], R1IsoNoOsc[1,:,:].T) + np.multiply(R1Fiss[2,:], R1IsoNoOsc[2,:,:].T) + np.multiply(R1Fiss[3,:], R1IsoNoOsc[3,:,:].T) 
    R2Spectrum = np.multiply(R2Fiss[0,:], R2IsoNoOsc[0,:,:].T) + np.multiply(R2Fiss[1,:], R2IsoNoOsc[1,:,:].T) + np.multiply(R2Fiss[2,:], R2IsoNoOsc[2,:,:].T) + np.multiply(R2Fiss[3,:], R2IsoNoOsc[3,:,:].T) 
    
    R1OscTotals = np.sum(R1OscSpectrum, axis=0) #(NumberOfRuns, )
    R2OscTotals = np.sum(R2OscSpectrum, axis=0)
    R1Totals = np.sum(R1Spectrum, axis=0)
    R2Totals = np.sum(R2Spectrum, axis=0)
    with np.errstate(invalid='ignore'):
        R1Norm = (Efficiency*Livetime)*np.divide(R1Bugey, R1Totals)
        R1Norm[np.isfinite(R1Norm) == False] = 0. 
        R2Norm = (Efficiency*Livetime)*np.divide(R2Bugey, R2Totals)
        R2Norm[np.isfinite(R2Norm) == False] = 0. 


    TotalSpectra = np.multiply(R1OscSpectrum, R1Norm) + np.multiply(R2OscSpectrum, R2Norm)
    Bump = Efficiency*Livetime*Prot*( np.outer(R1BumpSpectrum, R1FissionTotals) + np.outer(R2BumpSpectrum, R2FissionTotals)) #EnergyBins, NumberOfRuns, 
    
    TotalSpectra += Bump
    
    NonOscTotalSpectra = np.ones([TotalSpectra.size])
    return TotalSpectra, NonOscTotalSpectra
    
def GetFullSpectrum(Det, R1IsotopeOsc_, R2IsotopeOsc_, R1IsotopeNoOsc_, R2IsotopeNoOsc_, U235Flux, U238Flux, Pu239Flux, Pu241Flux, Bugey4, R1Mean, R2Mean, 
    Bugey4Fractions, R1Fissions, R2Fissions, Efficiency, LivetimeRatio, H, R1BumpSpectrum, R2BumpSpectrum):
    
    R1IsotopeOsc_[0, :, :] = np.multiply(R1IsotopeOsc_[0, :, :], U235Flux)
    R1IsotopeOsc_[1, :, :] = np.multiply(R1IsotopeOsc_[1, :, :], U238Flux)
    R1IsotopeOsc_[2, :, :] = np.multiply(R1IsotopeOsc_[2, :, :], Pu239Flux)
    R1IsotopeOsc_[3, :, :] = np.multiply(R1IsotopeOsc_[3, :, :], Pu241Flux)
    
    R2IsotopeOsc_[0, :, :] = np.multiply(R2IsotopeOsc_[0, :, :], U235Flux)
    R2IsotopeOsc_[1, :, :] = np.multiply(R2IsotopeOsc_[1, :, :], U238Flux)
    R2IsotopeOsc_[2, :, :] = np.multiply(R2IsotopeOsc_[2, :, :], Pu239Flux)
    R2IsotopeOsc_[3, :, :] = np.multiply(R2IsotopeOsc_[3, :, :], Pu241Flux)

    R1IsotopeNoOsc_[0, :, :] = np.multiply(R1IsotopeNoOsc_[0, :, :], U235Flux)
    R1IsotopeNoOsc_[1, :, :] = np.multiply(R1IsotopeNoOsc_[1, :, :], U238Flux)
    R1IsotopeNoOsc_[2, :, :] = np.multiply(R1IsotopeNoOsc_[2, :, :], Pu239Flux)
    R1IsotopeNoOsc_[3, :, :] = np.multiply(R1IsotopeNoOsc_[3, :, :], Pu241Flux)
    
    R2IsotopeNoOsc_[0, :, :] = np.multiply(R2IsotopeNoOsc_[0, :, :], U235Flux)
    R2IsotopeNoOsc_[1, :, :] = np.multiply(R2IsotopeNoOsc_[1, :, :], U238Flux)
    R2IsotopeNoOsc_[2, :, :] = np.multiply(R2IsotopeNoOsc_[2, :, :], Pu239Flux)
    R2IsotopeNoOsc_[3, :, :] = np.multiply(R2IsotopeNoOsc_[3, :, :], Pu241Flux)
    
    SpectrumOsc, Spectrum = GetFinalSpectrum(Det, Bugey4, Bugey4Fractions, R1Fissions, R2Fissions, 
        R1IsotopeOsc_, R2IsotopeOsc_, R1IsotopeNoOsc_, R2IsotopeNoOsc_, 
        R1Mean, R2Mean, Efficiency, LivetimeRatio, R1BumpSpectrum, R2BumpSpectrum)
        
    return np.dot(H, SpectrumOsc), Spectrum

def OrganizeEvents(Events, EventRunNumbers, File):
    F = np.loadtxt(File, unpack=True)
    LastEvent = 0
    NumberOfEvents = Events.size
    List = []
    Count = 0
    for RunNumber in F:
        Flag = True
        EventList = []
        RunNumber = int(RunNumber)
        for i in range(LastEvent, NumberOfEvents):
            ChosenNum = int(EventRunNumbers[i])
            if RunNumber == ChosenNum:
                EventList.append(Events[i])
                Flag = False
                LastEvent = i
                Count += 1
            else:
                if Flag == False:
                    break
        List.append(EventList)  
    if Count != NumberOfEvents:
        print "STOP ITERATION, NOT ALL EVENTS ARE LOADED" 
    
    return List
    
 
def VetoEvents(EventList, Veto):
    VetoedEventList = []
    VetoedEvents = []
    NumberOfSubLists = len(EventList)
    print NumberOfSubLists
    for i in range(NumberOfSubLists):
        if Veto[i] == 1.:
            if len(EventList[i]) == 0:
                VetoedEventList.append([])
                continue
            VetoedEventList.append(EventList[i])
            VetoedEvents.extend(EventList[i])
    return VetoedEventList, np.asarray(VetoedEvents)
        

@jit
def GetBinCenters(Edges):
    Centers = np.zeros([Edges.size-1])
    for i in range(Edges.size-1):
        dEdge = .5*(Edges[i+1]-Edges[i])
        Centers[i] = dEdge + Edges[i]
    return Centers

@jit
def GetEfficiencies(NumberOfSamples):
    
    if os.path.isfile("FarEff.npz"):
        print "Loading Archived Efficiencies."
        return np.load("FarEff.npy"), np.load("NewFarEff.npy"), np.load("NearEff.npy")
    
    
    SpillIn_SpillOut = np.random.normal(1.01350, 0.0027, NumberOfSamples)
    FarProtonNumberUncertainty = np.random.normal(1.0000, 0.0038, NumberOfSamples)
    NearProtonNumberUncertainty = np.random.normal(1.0000, 0.0037, NumberOfSamples)
    #FarGDFraction = [.8521, .8550, .8563, .8540, .8548]
    #FarGDFraction = [.8558, .8561, .8622, .8691, .8752] #5th Calibration
    FarGDFraction = [0.85363,  0.85759, 0.85644]
    GDMean = np.mean(FarGDFraction)
    GDSTD = np.std(FarGDFraction)
    ThirdPubGDFraction = np.random.normal(GDMean, GDSTD, NumberOfSamples)
    NearGDFractions = [0.84945,  0.85430,  0.85287]
    NDGDMean = np.mean(NearGDFractions)
    NDGDSigma = np.std(NearGDFractions)
    print ("Average Near GD Fraction: ", NDGDMean, " pm ", NDGDSigma)
    NDGDMean = .8541
    NDGDSigma = .03/100
    NearGDFaction = np.random.normal(NDGDMean, NDGDSigma, NumberOfSamples)
    print "Going with a GD Fraction of: ", NDGDMean, " pm ", NDGDSigma

    OldFarDetectionEfficiency = np.random.normal(0.99023, 0.00031, NumberOfSamples)
    FarDetectionEfficiency = np.random.normal(0.99023, 0.00031, NumberOfSamples)
    NearDetectionEfficiency= np.random.normal(0.98928, 0.00038, NumberOfSamples)

    FarDetectorLivetimeEff = np.random.normal(.9640, (.03/100.), NumberOfSamples)
    FarIIDetectorLivetimeEff = np.random.normal(.9586, (.05/100), NumberOfSamples)
    NearDetectorLivetimeEff = np.random.normal(.9602, (.02/100.), NumberOfSamples)

    ThirdPubEfficiencies = np.multiply(ThirdPubGDFraction, OldFarDetectionEfficiency)
    ThirdPubEfficiencies = np.multiply(ThirdPubEfficiencies, FarDetectorLivetimeEff)
    ThirdPubEfficiencies = np.multiply(ThirdPubEfficiencies, SpillIn_SpillOut)
    ThirdPubEfficiencies = np.multiply(ThirdPubEfficiencies, FarProtonNumberUncertainty)

    NewFarEfficiencies = np.multiply(ThirdPubGDFraction, FarDetectionEfficiency)
    NewFarEfficiencies = np.multiply(NewFarEfficiencies, FarIIDetectorLivetimeEff)
    NewFarEfficiencies = np.multiply(NewFarEfficiencies, SpillIn_SpillOut)
    NewFarEfficiencies = np.multiply(NewFarEfficiencies, FarProtonNumberUncertainty)

    NewNearEfficiencies = np.multiply(NearDetectionEfficiency, NearDetectorLivetimeEff)
    NewNearEfficiencies = np.multiply(NewNearEfficiencies, NearGDFaction)
    NewNearEfficiencies = np.multiply(NewNearEfficiencies, SpillIn_SpillOut) 
    NewNearEfficiencies = np.multiply(NewNearEfficiencies, NearProtonNumberUncertainty)
    OldVolume = np.pi*((2.301/2)**2)*2.455
    NewVolume = np.pi*1.*2.
    FiducialVolumeCut = NewVolume/OldVolume
    print ("Fiducial Volume Cut: ", FiducialVolumeCut)
    #NewNearEfficiencies *= FiducialVolumeCut
    
    np.save("FarEff", ThirdPubEfficiencies)
    np.save("NewFarEff", NewFarEfficiencies)
    np.save("NearEff", NewNearEfficiencies)
    
    return ThirdPubEfficiencies, NewFarEfficiencies, NewNearEfficiencies

def GetBackgroundHistograms(NumberOfSamples, VisibleBinEdges, Li9Histogram, OldFarAccHistogram, FarAccHistogram, NearAccHistogram):
    FNSMNorm = 0.
    Li9Histograms = []
    OldAccHistograms =[]
    AccHistograms = []
    NearAccHistograms = []
    FNHistograms = []
    NumberOfVisibleEnergyBins = len(VisibleBinEdges) - 1
    
    #FNMean = np.array([1.93101e+01, 1.72246e-01, -3.03470e-03])
    #FNCov = np.array([[3.7992, -.115082, .000810603], [-.115082, .00472601, -3.90807e-5], [.000810603, -3.90807e-5, 3.48574e-7]])

    FNMean = np.array([1., 0., 0.])
    FNCov = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def FNFunc(x, a, b, c):
        return a + b*x + c*x*x
        
    def FNFuncInt(x0, x1, a, b, c):
        #A = a*x0 + .5*b*x0*x0 + (1./3.)*c*x0*x0*x0
        #B = a*x1 + .5*b*x1*x1 + (1./3.)*c*x1*x1*x1
        B = x1
        A = x0
        return B-A
        
    for i in range(NumberOfSamples):
        A = np.zeros(NumberOfVisibleEnergyBins)
        B = np.zeros(NumberOfVisibleEnergyBins)
        C = np.zeros(NumberOfVisibleEnergyBins)
        D = np.zeros(NumberOfVisibleEnergyBins)
        FN = np.zeros(NumberOfVisibleEnergyBins)
        Li9Norm = 0.
        AccNorm = 0.
        NearAccNorm = 0.
        OldAccNorm = 0.
        FNCoeff = np.random.multivariate_normal(FNMean, FNCov)
        Total = FNFuncInt(VisibleBinEdges[0], VisibleBinEdges[NumberOfVisibleEnergyBins], FNCoeff[0], FNCoeff[1], FNCoeff[2])
        
        for j in range(NumberOfVisibleEnergyBins):
            dVisibleEnergy = VisibleBinEdges[j+1] - VisibleBinEdges[j]
            if Li9Histogram[j] != 0:
                A[j] = np.random.normal(Li9Histogram[j], np.sqrt(Li9Histogram[j]))
            if FarAccHistogram[j] != 0:
                B[j] = np.random.normal(FarAccHistogram[j], np.sqrt(FarAccHistogram[j]))
            if NearAccHistogram[j] != 0:
                C[j] = np.random.normal(NearAccHistogram[j], np.sqrt(NearAccHistogram[j]))
            if OldFarAccHistogram[j] != 0:
                D[j] = np.random.normal(OldFarAccHistogram[j], np.sqrt(OldFarAccHistogram[j]))
            
            FN[j] = FNFuncInt(VisibleBinEdges[j], VisibleBinEdges[j+1], FNCoeff[0], FNCoeff[1], FNCoeff[2])
            Li9Norm += A[j]*dVisibleEnergy
            AccNorm += B[j]*dVisibleEnergy
            NearAccNorm += C[j]*dVisibleEnergy
            OldAccNorm += D[j]*dVisibleEnergy
        A = A/Li9Norm
        B = B/AccNorm
        C = C/NearAccNorm
        D = D/OldAccNorm
        FN = FN/Total

        Li9Histograms.append(A)
        AccHistograms.append(B)
        NearAccHistograms.append(C)
        OldAccHistograms.append(D)
        FNHistograms.append(FN)
    
    return Li9Histograms, AccHistograms, NearAccHistograms, OldAccHistograms, FNHistograms


@jit
def PositronEnergy(NeutrinoEnergy):
    MDiff = -NeutrinoEnergy + Delta + (Delta * Delta - Me * Me)/(2.*Mp)
    EPos = Mn*Mn - 4.*Mp*MDiff
    EPos = .5*(np.sqrt(EPos) - Mn)
    return EPos

def GetTrueToVisibleConversion(NumberOfSamples, TrueEnergies, VisibleEnergies, TrueBins, VisibleBins):
    H, xedges, yedges = np.histogram2d(TrueEnergies, VisibleEnergies, bins=(TrueBins, VisibleBins))
    ListOfHs = []
    
    NumTrue = len(TrueBins)-1
    NumVis = len(VisibleBins)-1
    
    for i in range(NumberOfSamples):
        h = np.zeros([NumTrue, NumVis])
        hNorms = np.zeros([NumTrue])
        for j in range(NumTrue):
            hNorms[j] = 0.
            for k in range(NumVis):
                if H[j][k] == 0:
                    continue
                l = np.random.normal(H[j][k], np.sqrt(H[j][k]))
                while l <= 0.:
                    l = np.random.normal(H[j][k], np.sqrt(H[j][k]))
                h[j][k] = l
                hNorms[j] += h[j][k]

        for j in range(NumTrue):
            if hNorms[j] == 0.:
                continue
            for k in range(NumVis):
                h[j][k] = h[j][k]/hNorms[j]
           
        ListOfHs.append(h.T)
    return ListOfHs

@jit
def FluxFunction(E, C):
    Value = C[0] + C[1]*E + C[2]*E*E + C[3]*E*E*E
    Value += C[4]*E*E*E*E + C[5]*E*E*E*E*E
    return np.exp(Value)
   
@jit 
def PositronEnergy(NeutrinoEnergy):
    MDiff = -NeutrinoEnergy + Delta + (Delta * Delta - Me * Me)/(2.*Mp)
    EPos = Mn*Mn - 4.*Mp*MDiff
    EPos = .5*(np.sqrt(EPos) - Mn)
    return EPos

@jit
def IBDCrossSection(NeutrinoEnergy):
    EPos = PositronEnergy(NeutrinoEnergy)
    return np.where((( np.power(EPos, 2) - Me*Me > 0.) & (EPos > 0.)), K*EPos*np.sqrt(EPos*EPos - Me*Me), 0.)

''''
@jit
def IBDCrossSection(NeutrinoEnergy):
    E_Positron = NeutrinoEnergy - Delta
    P_Positron = np.sqrt(E_Positron*E_Positron - Me*Me)
    Arg = -0.07056 + 0.02018*np.log(NeutrinoEnergy) - 0.001953*np.power(np.log(NeutrinoEnergy), 3.)
    return (1.e-43)*P_Positron*E_Positron*np.power(NeutrinoEnergy, Arg)
'''
@jit
def InverseSquare(Baseline):
    return 1./(4.*Baseline*Baseline*100.*100.*np.pi)

@jit
def OscillationFunction(NeutrinoEnergy, Baseline, SSTT13, M13):
    M13 = np.fabs(M13)
    if NeutrinoEnergy == 0.:
        return 0.
    SineValue = np.sin(1.267*M13*Baseline/NeutrinoEnergy)
    Value = 1. - SSTT13*SineValue*SineValue
    return Value
    
@jit
def SpectrumSansFlux(Baseline, NeutrinoEnergy, SSTT13, M13):
    return OscillationFunction(NeutrinoEnergy, Baseline, SSTT13, M13)*InverseSquare(Baseline)*IBDCrossSection(NeutrinoEnergy)

@jit
def UnOscillatedSpectrumSansFlux(Baseline, NeutrinoEnergy): 
    return InverseSquare(Baseline)*IBDCrossSection(NeutrinoEnergy)
    
@jit
def IntegratedSpectrumSansFlux(NeutrinoEnergy, Reactor, SSTT13, M13, R1Lengths, R2Lengths, R1Hist, R2Hist):
    Sum = 0.
    if Reactor == 0:
        if NeutrinoEnergy > 12.:
            return 0.
        Spectra = SpectrumSansFlux(R1Lengths, NeutrinoEnergy, SSTT13, M13)
        Sum = np.dot(Spectra, R1Hist)*(R1Lengths[1]-R1Lengths[0])
    if Reactor == 1:
        if NeutrinoEnergy > 12.:
            return 0.
        Spectra = SpectrumSansFlux(R2Lengths, NeutrinoEnergy, SSTT13, M13)
        Sum = np.dot(Spectra, R2Hist)*(R2Lengths[1]-R2Lengths[0])
    return Sum

@jit
def IntegratedUnOscillatedSpectrumSansFlux(NeutrinoEnergy, Reactor, R1Lengths, R2Lengths, R1Hist, R2Hist):
    Sum = 0.
    if Reactor == 0:
        if NeutrinoEnergy > 12.:
            return 0.
        
        Spectra = InverseSquare(R1Lengths)*IBDCrossSection(NeutrinoEnergy)
        Sum = np.dot(Spectra, R1Hist)*(R1Lengths[1]-R1Lengths[0])
    if Reactor == 1:
        if NeutrinoEnergy > 12.:
            return 0.
        
        Spectra = InverseSquare(R2Lengths)*IBDCrossSection(NeutrinoEnergy)
        Sum = np.dot(Spectra, R2Hist)*(R2Lengths[1]-R2Lengths[0])
    return Sum