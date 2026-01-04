
#
# Check normalization of exported histograms
#

import ROOT


# Configuration
samples = [
    "p8_ee_Ztautau_ecm91",
    "p8_ee_Ztautau_Mtau_m1p0MeV_ecm91",
    "p8_ee_Ztautau_Mtau_m10p0MeV_ecm91",
    "p8_ee_Ztautau_Mtau_p1p0MeV_ecm91",
    "p8_ee_Ztautau_Mtau_p10p0MeV_ecm91",
    "p8_ee_Ztautau_Mnutau_10p0MeV_ecm91",
    "p8_ee_Ztautau_Mnutau_100p0MeV_ecm91",
]

# Percent of selected events
reductionFactors = [
    0.115810,
    0.115918,
    0.115756,
    0.115226,
    0.115633,
    0.115773,
    0.115370,
]

# Branch to count events over
branch = "jet1_mass"

# Cross section of the sample
crossSection = 1476.58 # pb

# Total integrated luminosity
luminosity = 205 * 1E+06 # /pb

print("")

for i in range(0, len(samples)):

    sampleFile = samples[i]

    # Load histograms from ROOT files
    filePath = f"outputs/histmaker/Ztautau_sens/{sampleFile}.root"
    rootFile = ROOT.TFile.Open(filePath)
    hist = rootFile.Get(branch)

    Nobs = 0
    for j in range(0, hist.GetNbinsX() + 1):
        Nobs += hist.GetBinContent(j)

    # Assuming that all events have the same weight
    Nexp = crossSection * luminosity * reductionFactors[i]

    print(f"{sampleFile}: \t Diff:{Nobs - Nexp} \t Ratio:{Nobs/Nexp}")

    rootFile.Close()

print("")
