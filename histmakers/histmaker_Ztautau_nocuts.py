
#
# Hist maker for Ztautau mass sensibility study
#

import getpass

# Center of mass energy
ecm = 91

# Default binning
binsJet1Mass = (100, 0.0, 3)
binsJetsTotalMass = (1000, 0.0, 100.0)
binsDiscreteVar = (10, 0.0, 10.0)
binsAngleVar = (100, 0.0, 3.14)

# List of processes
processList = {
	'p8_ee_Ztautau_ecm91': {},
}


# Codename given to the analysis
analysisName = "reco_nocuts"

# Channel ("hadronic", "semihadronic", "leptonic")
channel = "hadronic"


# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
#prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictonary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Define the input dir
username = getpass.getuser()
inputDir = f"/eos/user/{username[0]}/{username}/Ztautau/{analysisName}/{channel}"

#Optional: output directory, default is local running directory
outputDir = f"./outputs/Ztautau_{analysisName}/"

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

    # Leptons

    histNlep = df.Histo1D(("nlep", "", *binsDiscreteVar), "nlep")
    results.append(histNlep)

    histLepMomentum = df.Histo1D(("lep_p", "", *binsJetsTotalMass), "lep_p")
    results.append(histLepMomentum)

    histLepTheta = df.Histo1D(("lep_theta", "", *binsAngleVar), "lep_theta")
    results.append(histLepTheta)

    histLepPhi = df.Histo1D(("lep_phi", "", *binsAngleVar), "lep_phi")
    results.append(histLepPhi)

    histEleIsolation = df.Histo1D(("electrons_isolation", "", *binsJetsTotalMass), "electrons_isolation")
    results.append(histEleIsolation)

    histMuonIsolation = df.Histo1D(("muons_isolation", "", *binsJetsTotalMass), "muons_isolation")
    results.append(histMuonIsolation)

    # Jets

    histNjets = df.Histo1D(("njets", "", *binsDiscreteVar), "njets")
    results.append(histNjets)

    histNprongs = df.Histo1D(("nprongs", "", *binsDiscreteVar), "nprongs")
    results.append(histNprongs)

    histJet1TotalCharge = df.Histo1D(("jet1_total_charge", "", *binsDiscreteVar), "jet1_total_charge")
    results.append(histJet1TotalCharge)

    histJet2TotalCharge = df.Histo1D(("jet2_total_charge", "", *binsDiscreteVar), "jet2_total_charge")
    results.append(histJet2TotalCharge)

    histJet1Mass = df.Histo1D(("jet1_mass", "", *binsJet1Mass), "jet1_mass")
    results.append(histJet1Mass)

    histJet2Mass = df.Histo1D(("jet2_mass", "", *binsJet1Mass), "jet2_mass")
    results.append(histJet2Mass)

    histJetsTotalMass = df.Histo1D(("jets_total_mass", "", *binsJetsTotalMass), "jets_total_mass")
    results.append(histJetsTotalMass)

    histJet1Theta = df.Histo1D(("jet1_theta", "", *binsAngleVar), "jet1_theta")
    results.append(histJet1Theta)

    histJet1Phi = df.Histo1D(("jet1_phi", "", *binsAngleVar), "jet1_phi")
    results.append(histJet1Phi)

    histJet2Theta = df.Histo1D(("jet2_theta", "", *binsAngleVar), "jet2_theta")
    results.append(histJet2Theta)

    histJet2Phi = df.Histo1D(("jet2_phi", "", *binsAngleVar), "jet2_phi")
    results.append(histJet2Phi)

    return results, weightsum
