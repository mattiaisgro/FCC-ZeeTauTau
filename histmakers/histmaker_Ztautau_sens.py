
#
# Hist maker for Ztautau mass sensibility study
#

import getpass

# Center of mass energy
ecm = 91

# Default binning
binsJetMass = (100, 1, 2)

# List of processes
processList = {
    'p8_ee_Ztautau_ecm91': {},
    #'p8_ee_Ztautau_Mtau_m1p0MeV_ecm91': {
    #    "crossSection": 1476.58, # pb
    #},
    #'p8_ee_Ztautau_Mtau_p1p0MeV_ecm91': {
    #    "crossSection": 1476.58, # pb
    #},
    #'p8_ee_Ztautau_Mtau_m5p0MeV_ecm91': {
    #    "crossSection": 1476.58, # pb
    #},
    #'p8_ee_Ztautau_Mtau_p5p0MeV_ecm91': {
    #    "crossSection": 1476.58, # pb
    #},
    'p8_ee_Ztautau_Mtau_m10p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
    'p8_ee_Ztautau_Mtau_p10p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
    'p8_ee_Ztautau_Mnutau_10p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
    'p8_ee_Ztautau_Mnutau_50p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
    'p8_ee_Ztautau_Mnutau_100p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
    'p8_ee_Ztautau_Mnutau_200p0MeV_ecm91': {
        "crossSection": 1476.58, # pb
    },
}


# Codename given to the analysis
analysisName = "sens_notagger_3prongs"

# Channel (hadronic, semihadronic, leptonic)
channel = "hadronic"


# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Define the input dir
username = getpass.getuser()
inputDir = f"/eos/user/{username[0]}/{username}/Ztautau/reco_notagger/{channel}"

# Optional: output directory, default is local running directory
outputDir = f"./outputs/histmaker/Ztautau_{analysisName}/"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = 4

# scale the histograms with the cross-section and integrated luminosity
doScale = True

# Integrated Luminosity = 205 /ab
intLumi = 205000000 # /pb


def build_graph(df, dataset):

    results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    df = df.Filter("nprongs[0] == 3")

    histJet1Mass = df.Histo1D(("jet1_mass", "", *binsJetMass), "jet1_mass")
    results.append(histJet1Mass)

    # Cut out failed reconstructions
    #df2 = df.Filter("jets_reco_mass != 0")
    #histJetsRecoMass = df2.Histo1D(("jets_reco_mass", "", *binsJetMass), "jets_reco_mass")
    #results.append(histJetsRecoMass)

    return results, weightsum
