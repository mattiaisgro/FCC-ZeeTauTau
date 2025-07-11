
import getpass
import os
import ROOT
import uproot
import awkward


# The process under study
process = "Ztautau/selection"


# The 2 branches to draw
branch_x = "jets_total_mass"
branch_y = "nprongs"

# Histogram settings
nbins_x, xmin, xmax = 100, 0, 100
nbins_y, ymin, ymax = 6, 0, 6

histName = "hist2d"
histTitle = "Jets Invariant Mass vs. N. Prongs;Single Jet Invariant Mass (GeV);N. Prongs"


# Input and output configuration
processFilename = process.replace("/", "_")
username = getpass.getuser()
inputFolder = f"/eos/user/{username[0]}/{username}/{process}/hadronic/p8_ee_Ztautau_ecm91/"
outputFile = f"./plots/{processFilename}_{branch_x}_{branch_y}.png"


# Limit the number of chunks read (for trial-and-error)
maxFiles = 99999


# Given a list of directories, list the ROOT files in it.
def get_file_list(directory):

	files = []
	root_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".root")]
	files.extend(root_files)

	return files[:maxFiles]


# Fill each histogram with data, given the list of files and the branch name
def build_hist(files):

	hist2d = ROOT.TH2F(histName, histTitle, nbins_x, xmin, xmax, nbins_y, ymin, ymax)

	for file in files:

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
				branchesData = tree.arrays([branch_x, branch_y], how=dict)
			except Exception as e:
				print(f"Some branches were not found or error occurred: {e}, skipping chunk...")
				continue

			x_data = awkward.to_numpy(branchesData[branch_x])
			y_data = awkward.to_numpy(branchesData[branch_y])

			for i in range(0, len(x_data)):

				# Only select events with 2-jets
				# with an equal number of prongs
				#if y_data[i][0] != y_data[i][1]:
				#	continue

				hist2d.Fill(x_data[i], y_data[i][0])

	return hist2d


# Load data and construct the histogram
files = get_file_list(inputFolder)
hist2d = build_hist(files)

# Draw and save with improved resolution and aesthetics
# Create a canvas with higher resolution
c = ROOT.TCanvas("c", "c", 1600, 1200)
c.SetRightMargin(0.15)
c.SetLeftMargin(0.12)
c.SetBottomMargin(0.12)
c.SetTopMargin(0.08)

# Set logarithmic scale on Z axis (event count)
c.SetLogz()

# Improve histogram aesthetics
hist2d.SetStats(0)
hist2d.SetTitle(histTitle)
hist2d.GetXaxis().SetTitleSize(0.045)
hist2d.GetYaxis().SetTitleSize(0.045)
hist2d.GetXaxis().SetLabelSize(0.04)
hist2d.GetYaxis().SetLabelSize(0.04)
hist2d.GetZaxis().SetLabelSize(0.04)
hist2d.GetZaxis().SetTitleSize(0.045)
hist2d.GetZaxis().SetTitleOffset(1.1)

# Use a better color palette
ROOT.gStyle.SetPalette(ROOT.kBird)

hist2d.Draw("COLZ")

# Save as high-resolution PNG
c.SaveAs(outputFile)
print(f"2D histogram saved as {outputFile}")
