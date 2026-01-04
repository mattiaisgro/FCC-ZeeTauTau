
import os
import uproot
import ROOT
import getpass
import awkward


# Name of the process under study
process = "Ztautau/reco_njets"

# ID of the sample under study
sampleId = "p8_ee_Ztautau_ecm91"

# The channel to consider ("hadronic", "semihadronic", "leptonic")
channel = "hadronic"

# Branches (variables) to plot
branches = [
	#"jet1_mass",
	#"jet2_mass",
	"jets_total_mass",
	#"nprongs",
	#"jets_reco_mass",
	#"njets",
]

# Dictionary of plot titles, by branch name
title = {
	"jet1_mass": "Jet Invariant Mass (GeV)",
	"jet2_mass": "Jet Invariant Mass (GeV)",
	"jets_total_mass": "Jets Total Invariant Mass (GeV)",
	"nprongs": "Number of Prongs in 2-Jet Decays",
	"jets_reco_mass": "Reconstructed Jet Mass (GeV)",
	"njets": "Number of Jets",
}

# Range of the variables in the histogram
histRanges = {
	"jet1_mass": (0.01, 2),
	"jet2_mass": (0, 2),
	"jets_total_mass": (0, 100),
	"nprongs": (0, 6),
	"jets_reco_mass": (0.01, 2),
	"njets": (2.0, 8.0),
}

# List of signal directories
localUsername = getpass.getuser()
signalDirs = [
	f"/eos/user/{localUsername[0]}/{localUsername}/{process}/{channel}/{sampleId}/"
]

signalLabel = "Z #rightarrow #tau #tau"
signalScale = 1.0

# List of background directories
backgroundDirs = [
	#f"/eos/user/{localUsername[0]}/{localUsername}/{process}/{channel}/p8_ee_Zud_ecm91/",
	#f"/eos/user/{localUsername[0]}/{localUsername}/{process}/{channel}/p8_ee_Zcc_ecm91/",
	#f"/eos/user/{localUsername[0]}/{localUsername}/{process}/{channel}/p8_ee_Zss_ecm91/",
	#f"/eos/user/{localUsername[0]}/{localUsername}/{process}/{channel}/p8_ee_Zbb_ecm91/",
]

backgroundLabel = "Z #rightarrow q #bar q"
backgroundScale = 1.0

# Plot output directory
outputDir = "./"

# Number of bins for the histograms
histBins = 100

# Limit the number of chunks read (for trial-and-error)
maxFiles = 9999

# Bounds for the y axis
yMin, yMax = 1e-02, 1e+08


# Given a list of directories, list the ROOT files in it.
def get_file_list(directories):

	files = []
	for directory in directories:
		root_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".root")]
		files.extend(root_files)

	return files[:maxFiles]


# Fill each histogram with data, given the list of files and the branch name
def build_hist(files, branches):

	global histBins
	global histRanges

	hists = {}

	for branch in branches:
		hists[branch] = ROOT.TH1F("", f"{title[branch]}", histBins, *histRanges[branch])

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
			#print("Available Branches = " + str(tree.keys()))
			
			try:
				branchesData = tree.arrays(branches, how=dict)
			except Exception as e:
				print(f"Some branches were not found or error occurred: {e}, skipping chunk...")
				continue

			for branch in branches:

				data = awkward.to_numpy(branchesData[branch])
				#data = awkward.to_numpy(awkward.flatten(branchesData[branch]))
				for value in data:
					hists[branch].Fill(value)

	return hists


print("Collecting signal files...")
signalFiles = get_file_list(signalDirs)
print(f"Found {len(signalFiles)} signal files.")

print("Collecting background files...")
backgroundFiles = get_file_list(backgroundDirs)
print(f"Found {len(backgroundFiles)} background files.")


print("Building signal histograms...")
signalHists = build_hist(signalFiles, branches)

print("Building background histograms...")
backgroundHists = build_hist(backgroundFiles, branches)


# Save histograms
canvas = ROOT.TCanvas("canvas", "", 1600, 1200)

# Set left and bottom margins to avoid axis title cutoff
canvas.SetLeftMargin(0.17)
canvas.SetBottomMargin(0.17)

for branch in branches:
	print(f"Plotting histograms for branch: {branch}")

	signalHist = signalHists[branch]
	backgroundHist = backgroundHists[branch]

	# Set line and fill colors for clarity
	signalHist.SetLineColor(ROOT.kRed)
	signalHist.SetLineWidth(3)
	signalHist.SetFillColorAlpha(ROOT.kRed, 0.35)

	backgroundHist.SetLineColor(ROOT.kBlue + 2)
	backgroundHist.SetLineWidth(3)
	backgroundHist.SetFillColorAlpha(ROOT.kBlue + 2, 0.35)

	# Improve axis label font and size
	signalHist.GetXaxis().SetTitleFont(42)
	signalHist.GetYaxis().SetTitleFont(42)
	signalHist.GetXaxis().SetLabelFont(42)
	signalHist.GetYaxis().SetLabelFont(42)
	signalHist.GetXaxis().SetTitleSize(0.05)
	signalHist.GetYaxis().SetTitleSize(0.05)
	signalHist.GetXaxis().SetLabelSize(0.045)
	signalHist.GetYaxis().SetLabelSize(0.045)

	# Increase title offsets to prevent cutoff
	signalHist.GetXaxis().SetTitleOffset(1.4)
	signalHist.GetYaxis().SetTitleOffset(1.6)

	# Normalize histograms if desired (optional)
	signalHist.Scale(signalScale)
	backgroundHist.Scale(backgroundScale)

	# Set logarithmic scale for y axis
	canvas.SetLogy(True)

	# Set axis labels
	signalHist.GetXaxis().SetTitle(title[branch])
	signalHist.GetYaxis().SetTitle("Events")
	#signalHist.SetMaximum(yMax)
	#signalHist.SetMinimum(yMin)

	# Draw both histograms on the same canvas
	signalHist.Draw("HIST")
	#backgroundHist.Draw("HIST SAME")

	# Add legend
	signalHist.SetStats(0)
	backgroundHist.SetStats(0)

	legend = ROOT.TLegend(0.76, 0.80, 0.90, 0.90)
	legend.SetMargin(0.18)
	legend.AddEntry(signalHist, signalLabel, "f")
	#legend.AddEntry(backgroundHist, backgroundLabel, "f")
	legend.SetTextSize(0.035)
	legend.Draw()

	outputFile = f"{outputDir + process.replace('/', '_')}_{branch}_{sampleId}.png"
	canvas.SaveAs(outputFile)
	print(f"Saved plot to {outputDir + outputFile}")
