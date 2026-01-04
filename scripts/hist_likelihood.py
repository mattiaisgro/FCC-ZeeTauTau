
import math
from multiprocessing import reduction
import ROOT


# Configuration
samples = [
    #"Mtau_m1p0MeV",
    #"Mtau_p1p0MeV",
    "Mtau_m5p0MeV",
    "Mtau_p5p0MeV",
    "Mtau_m10p0MeV",
    "Mtau_p10p0MeV",

    #"Mnutau_0p1MeV",
    #"Mnutau_1p0MeV",
    #"Mnutau_5p0MeV",
    #"Mnutau_10p0MeV",
    #"Mnutau_50p0MeV",
    #"Mnutau_100p0MeV",
    #"Mnutau_200p0MeV",
]

branches = [
    "jet1_mass",
    #"jets_reco_mass"
]

# Codename for the analysis
analysisName = "sens_5prongs"


for branch in branches:

    for sampleId in samples:

        #print(f"Processing {branch} of sample {sampleId}...")

        # File paths
        filePath1 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_ecm91.root"
        filePath2 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_{sampleId}_ecm91.root"
        histName = branch

        # Load histograms from ROOT files
        file1 = ROOT.TFile.Open(filePath1)
        file2 = ROOT.TFile.Open(filePath2)

        histCentral = file1.Get(histName)
        histVariation = file2.Get(histName)

        # Compute the log likelihood
        logL = 0
        for i in range(0, histCentral.GetNbinsX() + 1):

            Nobs = histCentral.GetBinContent(i)
            Nexp = histVariation.GetBinContent(i)
            #print(f"Nobs = {Nobs}, Nexp = {Nexp}")

            if Nobs != 0 and Nexp != 0:
                logL += -2 * (Nobs * (math.log(Nexp / Nobs)) + (Nobs - Nexp))

        print(f"{branch}\t{sampleId}\tLog Likelihood:\t{logL}")


        # Clean up
        file1.Close()
        file2.Close()

    print("")
