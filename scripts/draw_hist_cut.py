
import getpass
import uproot
import os
import ROOT
import numpy as np

# Configuration
nProngs = 5
process = "Ztautau/selection"

cutBranch = "nprongs"
plotBranch = "jet1_mass"

# Input and output
username = getpass.getuser()
processFilename = process.replace("/", "_")
inputFolder = f"/eos/user/{username[0]}/{username}/{process}/hadronic/p8_ee_Ztautau_ecm91/"
outputFile = f"./plots/{processFilename}_{plotBranch}_filter_{cutBranch}{nProngs}.png"


# Histogram configuration
histBins = 100
histRange = (0, 2)
histTitle = f"Single Jet Invariant Mass for N. Prongs = {nProngs}"


# Limit the number of chunks read (for trial-and-error)
maxFiles = 9999


# Given a list of directories, list the ROOT files in it.
def get_file_list(directory):

	files = []
	root_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".root")]
	files.extend(root_files)

	return files[:maxFiles]

hist = ROOT.TH1F("", histTitle, histBins, *histRange)

# Load target branches from files
files = get_file_list(inputFolder)

for file in files:

	with uproot.open(file) as rootFile:

		print("Loading " + file)
		
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
			branchesData = tree.arrays([plotBranch, cutBranch], how=dict)
		except Exception as e:
			print(f"Some branches were not found or error occurred: {e}, skipping chunk...")
			continue

		# Convert to numpy arrays for efficient masking
		plotValues = np.array(branchesData[plotBranch])
		cutValues = np.array(branchesData[cutBranch])

		# Mask: select events with 2 jets and equal number of prongs
		mask = (cutValues[:, 0] == cutValues[:, 1]) & (cutValues[:, 0] == nProngs)

		# Fill histogram with masked plot values
		for value in plotValues[mask]:
			hist.Fill(value)


# Save histograms
canvas = ROOT.TCanvas("canvas", "", 1600, 1200)

# Set left and bottom margins to avoid axis title cutoff
canvas.SetLeftMargin(0.17)
canvas.SetBottomMargin(0.17)

# Set line and fill colors for clarity
hist.SetLineColor(ROOT.kRed)
hist.SetLineWidth(3)
hist.SetFillColorAlpha(ROOT.kRed, 0.35)

# Improve axis label font and size
hist.GetXaxis().SetTitleFont(42)
hist.GetYaxis().SetTitleFont(42)
hist.GetXaxis().SetLabelFont(42)
hist.GetYaxis().SetLabelFont(42)
hist.GetXaxis().SetTitleSize(0.05)
hist.GetYaxis().SetTitleSize(0.05)
hist.GetXaxis().SetLabelSize(0.045)
hist.GetYaxis().SetLabelSize(0.045)

# Increase title offsets to prevent cutoff
hist.GetXaxis().SetTitleOffset(1.4)
hist.GetYaxis().SetTitleOffset(1.6)

# Normalize histograms if desired (optional)
hist.Scale(1.0)

# Set logarithmic scale for y axis
canvas.SetLogy(False)

# Set axis labels
hist.GetXaxis().SetTitle("Single Jet Invariant Mass (GeV)")
hist.GetYaxis().SetTitle("Events")
#hist.SetMaximum(yMax)
#hist.SetMinimum(yMin)

# Draw both histograms on the same canvas
hist.Draw("HIST")
hist.Draw("HIST SAME")

# Add legend
hist.SetStats(0)
hist.SetStats(0)

canvas.SaveAs(outputFile)
print(f"Saved plot to {outputFile}")

