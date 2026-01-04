
#
# Hist maker for Ztautau mass sensibility study
#

import getpass

# Center of mass energy
ecm = 91

# Default binning
binsJet1Mass = (100, 0.5, 2.5)
binsJetsTotalMass = (100, 0.0, 100.0)

# List of processes
processList = {
	'p8_ee_Ztautau_ecm91': {
		'fraction': 1,
		'chunks': 100,
	},
	'p8_ee_Zud_ecm91': {
		'fraction': 1,
		'chunks': 400,
	},
	'p8_ee_Zcc_ecm91': {
		'fraction': 1,
		'chunks': 400,
	},
	'p8_ee_Zss_ecm91': {
		'fraction': 1,
		'chunks': 400,
	},
	'p8_ee_Zbb_ecm91': {
		'fraction': 1,
		'chunks': 400,
	},
}


# Codename given to the analysis
analysisName = "reco_new"

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

    histJet1Mass = df.Histo1D(("jet1_mass", "", *binsJet1Mass), "jet1_mass")
    results.append(histJet1Mass)

    histJetsTotalMass = df.Histo1D(("jets_total_mass", "", *binsJetsTotalMass), "jets_total_mass")
    results.append(histJetsTotalMass)

    return results, weightsum
