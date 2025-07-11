
import os
import uproot
import ROOT
import getpass
import awkward
import math
import array


# Name of the process under study
process = "Ztautau/mass_sens_full"

# The channel to consider ("hadronic", "semihadronic", "leptonic")
channel = "hadronic"

# Target branch to plot
branch = "jets_reco_mass"
branchTitle = "Reco. Jet Inv. Mass (GeV)"

# Display name of the changed mass sample
sampleId = "Mnutau_1p0MeV"
sampleFolder = f"p8_ee_Ztautau_{sampleId}_ecm91"


# Input folder for "numerator" and "denominator" files
username = getpass.getuser()
numInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/p8_ee_Ztautau_ecm91"
denInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/{sampleFolder}"
outputFolder = "./plots/"

# Histogram options
histTitle = f"Ratio of Reco. Jet Invariant Mass (Central/{sampleId})"
histRange = (0.2, 3)
histBins = 500

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
	global histRanges

	hist = ROOT.TH1F("", f"{histTitle}", histBins, *histRange)

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
				branchesData = tree.arrays([branch], how=dict)
			except Exception as e:
				print(f"Some branches were not found or error occurred: {e}, skipping chunk...")
				continue

			data = awkward.to_numpy(branchesData[branch])
			#data = awkward.to_numpy(awkward.flatten(branchesData[branch]))
			for value in data:
				hist.Fill(value)

	return hist


numFiles = get_file_list(numInputFolder)
print(f"Found {len(numFiles)} numerator files.")

denFiles = get_file_list(denInputFolder)
print(f"Found {len(denFiles)} denominator files.")


print("Building numerator histograms...")
numHist = build_hist(numFiles, branch)

print("Building denominator histograms...")
denHist = build_hist(denFiles, branch)


# Create the ratio histogram with the same binning as numHist
ratioHist = numHist.Clone("ratioHist")
ratioHist.SetTitle(histTitle)

for i in range(1, numHist.GetNbinsX() + 1):

	num = numHist.GetBinContent(i)
	den = denHist.GetBinContent(i)

	if den != 0:
		ratio = num / den
	else:
		ratio = 0

	ratioHist.SetBinContent(i, ratio)


# Save histogram to file
print(f"Plotting ratio histogram for branch: {branch}")
canvas = ROOT.TCanvas("canvas", "", 1600, 1200)

# Set left and bottom margins to avoid axis title cutoff
canvas.SetLeftMargin(0.17)
canvas.SetBottomMargin(0.17)

# Set line and fill colors for clarity
ratioHist.SetLineColor(ROOT.kBlue)
ratioHist.SetLineWidth(3)
ratioHist.SetFillColorAlpha(ROOT.kBlue, 0.35)

# Improve axis label font and size
ratioHist.GetXaxis().SetTitleFont(42)
ratioHist.GetYaxis().SetTitleFont(42)
ratioHist.GetXaxis().SetLabelFont(42)
ratioHist.GetYaxis().SetLabelFont(42)
ratioHist.GetXaxis().SetTitleSize(0.05)
ratioHist.GetYaxis().SetTitleSize(0.05)
ratioHist.GetXaxis().SetLabelSize(0.045)
ratioHist.GetYaxis().SetLabelSize(0.045)

# Increase title offsets to prevent cutoff
ratioHist.GetXaxis().SetTitleOffset(1.4)
ratioHist.GetYaxis().SetTitleOffset(1.6)

# Normalize histograms if desired (optional)
#ratioHist.Scale(1.0)

# Set logarithmic scale for y axis
canvas.SetLogy(False)

# Set axis labels
ratioHist.GetXaxis().SetTitle(branchTitle)
ratioHist.GetYaxis().SetTitle("Event Ratio")
#ratioHist.SetMaximum(yMax)
#ratioHist.SetMinimum(yMin)

# Draw both histograms on the same canvas
ratioHist.Draw("HIST")

# Add legend
ratioHist.SetStats(0)

outputFile = f"{outputFolder + process.replace('/', '_')}_ratio_{branch}_{sampleId}.png"
canvas.SaveAs(outputFile)
print(f"Saved plot to {outputFile}")


# Compute the uncertainties over bin counts and
# plot relevant statistics
chi_sqr = 0
ratios = []
sigmas = []

for i in range(1, numHist.GetNbinsX() + 1):

	N1 = numHist.GetBinContent(i)
	N2 = denHist.GetBinContent(i)

	if N2 == 0:
		ratio = 0
	else:
		ratio = N1 / N2

	# sigma^2 = (N1 / N2^2) + (N1^2 / N2^3)
	if N2 == 0:
		sigma = 0
	else:
		sigma = math.sqrt(ratio * (1 + ratio) / N2)

	ratios.append(ratio)
	sigmas.append(sigma)
	if N2 != 0:
		chi_sqr += ((ratio - 1) / sigma)**2


# Compute the vector of central bin points from histRange and histBins
power = 2
bin_width = (histRange[1] - histRange[0]) / histBins
x_values = [histRange[0] + (i + 0.5) * bin_width for i in range(histBins)]
y_values = [((r - 1) / s)**power if s != 0 else 0 for r, s in zip(ratios, sigmas)]

# Plot x_values vs y_values using ROOT
x_arr = array.array('d', x_values)
y_arr = array.array('d', y_values)
graph = ROOT.TGraph(histBins, x_arr, y_arr)

graph.SetTitle(f"Norm. Sqr. Deviation per Bin ({sampleId});Reco. Jet Inv. Mass (GeV);Sqr. Norm. Deviation")
graph.SetLineColor(ROOT.kRed)
graph.SetLineWidth(2)

c2 = ROOT.TCanvas("c2", f"Norm. Sqr. Deviation ({sampleId})", 1600, 1200)
c2.SetLeftMargin(0.17)
c2.SetBottomMargin(0.17)
graph.Draw("AL")

# Add chi-squared values on the uncertainty plot
chi2_text = f"#chi^2 = {chi_sqr:.3f},  #chi^2/ndf = {chi_sqr/(histBins-1):.3f}"
latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextFont(42)
latex.SetTextSize(0.04)
latex.SetTextColor(ROOT.kBlack)
latex.DrawLatex(0.22, 0.85, chi2_text)
c2.Update()

outputFile2 = f"{outputFolder + process.replace('/', '_')}_uncertainty_{branch}_{sampleId}.png"
c2.SaveAs(outputFile2)
print(f"Saved uncertainty plot to {outputFile2}")

print("Chi-Squared = " + str(chi_sqr))
print("Norm. Chi-Squared = " + str(chi_sqr / (histBins - 1)))
