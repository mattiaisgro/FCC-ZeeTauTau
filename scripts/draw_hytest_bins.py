
import os
import uproot
import ROOT
import getpass
import awkward
import matplotlib.pyplot as plt

# Name of the process under study
process = "Ztautau/reco_new"

# The channel to consider ("hadronic", "semihadronic", "leptonic")
channel = "hadronic"

# Target branch to plot
branch = "jet1_mass"
branchTitle = "Reco. Jet Inv. Mass (GeV)"

# Display name of the changed mass sample
sampleId = "Mtau_m10p0MeV"
sampleFolder = f"p8_ee_Ztautau_{sampleId}_ecm91"


# Input folder for "numerator" and "denominator" files
username = getpass.getuser()
numInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/p8_ee_Ztautau_ecm91"
denInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/{sampleFolder}"
outputFolder = "./plots/"

# Histogram options
histTitle = f"Ratio of Reco. Jet Invariant Mass (Central/{sampleId})"
histRange = (0.2, 3)
histBins = [
	100,
	250,
	500,
	750,
	1000,
	1250,
	1500,
	1750,
	2000,
]

# Limit the number of chunks read (for trial-and-error)
maxFiles = 99999


# Given a list of directories, list the ROOT files in it.
def get_file_list(directory):

	files = []
	root_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".root")]
	files.extend(root_files)

	return files[:maxFiles]


# Fill each histogram with data, given the list of files and the branch name
def build_hist(files, branch):

	global histBins
	global histRange

	hists = []

	for bins in histBins:

		hist = ROOT.TH1F("", f"{histTitle}", bins, *histRange)

		for file in files:

			# print("Loading file: " + file)
			with uproot.open(file) as rootFile:
				
				# Find the correct tree key (handle possible cycle numbers)
				tree_key = None
				for key in rootFile.keys():
					if key.startswith("events"):
						tree_key = key
						break

				if tree_key is None:
					print(f"No 'events' tree found in file {file}, skipping...")
					continue

				tree = rootFile[tree_key]
				
				try:
					branchesData = tree.arrays([branch], how=dict)
				except Exception as e:
					print(f"Some branches were not found or error occurred: {e}, skipping chunk...")
					continue

				data = awkward.to_numpy(branchesData[branch])
				for value in data:
					hist.Fill(value)
			
		hists.append(hist)

	return hists


numFiles = get_file_list(numInputFolder)
print(f"Found {len(numFiles)} numerator files.")

denFiles = get_file_list(denInputFolder)
print(f"Found {len(denFiles)} denominator files.")


print("Building numerator histograms...")
numHists = build_hist(numFiles, branch)

print("Building denominator histograms...")
denHists = build_hist(denFiles, branch)

ksResults = []
for i in range(0, len(histBins)):
	ksProbability = numHists[i].KolmogorovTest(denHists[i], "X")
	ksResults.append(ksProbability)

# Plot ksResults against histBins
print("Plotting KS Results against Bins...")
plt.figure(figsize=(8, 5))
plt.plot(histBins, ksResults, marker='o', linestyle='-')
plt.xlabel('N. of Bins')
plt.ylabel('KS Test Probability')
plt.title('KS Test Probability vs. N. of Bins')
plt.grid(True)
plt.tight_layout()
processFilename = process.replace("/", "_")
plt.savefig(os.path.join(outputFolder, f"kstest_{processFilename}_{branch}_{sampleId}.png"))
