
import os
import uproot
import ROOT
import getpass
import awkward
import math
import array


# Name of the process under study
process = "Ztautau/reco_new"

# The channel to consider ("hadronic", "semihadronic", "leptonic")
channel = "hadronic"

# Target branch to plot
#branch = "jet1_mass"
branch = "jets_reco_mass"
branchTitle = "Reco. Jet Inv. Mass (GeV)"

# Samples to analyse
samples = [
	'Mnutau_0p1MeV',
	'Mnutau_1p0MeV',
	'Mnutau_10p0MeV',
	'Mnutau_100p0MeV',
	#'Mtau_p0p1MeV',
	#'Mtau_m0p1MeV',
	#"Mtau_m1p0MeV",
	#"Mtau_p1p0MeV",
	#"Mtau_m10p0MeV",
	#"Mtau_p10p0MeV",
]


for sampleId in samples:

	print("Analysing sample " + sampleId + "...")
	sampleFolder = f"p8_ee_Ztautau_{sampleId}_ecm91"

	# Input folder for "numerator" and "denominator" files
	username = getpass.getuser()
	numInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/p8_ee_Ztautau_ecm91"
	denInputFolder = f"/eos/user/{username[0]}/{username}/{process}/{channel}/{sampleFolder}"
	outputFolder = "./plots/"

	# Histogram options
	histTitle = f"Ratio of Reco. Jet Invariant Mass (Central/{sampleId})"
	histRange = (1, 2.5)
	histBins = 1000

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


	print("Computing KS Test Probability...")
	ksProbability = numHist.KolmogorovTest(denHist, "X")
	print("Kolmogorov-Smirnov probability: " + str(ksProbability))

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
	canvas = ROOT.TCanvas("canvas"+sampleId, "", 1600, 1200)

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
	ratioHist.SetMaximum(2)

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
			chi_sqr += ((ratio - 1) / sigma)**2 if sigma != 0 else 0


	# Compute the vector of central bin points from histRange and histBins
	bin_width = (histRange[1] - histRange[0]) / histBins
	x_values = [histRange[0] + (i + 0.5) * bin_width for i in range(histBins)]
	sqr_deviations = [((r - 1) / s)**2 if s != 0 else 0 for r, s in zip(ratios, sigmas)]
	deviations = [((r - 1) / s) if s != 0 else 0 for r, s in zip(ratios, sigmas)]


	# Plot x_values vs sqr_deviations using ROOT
	x_arr = array.array('d', x_values)
	y_arr = array.array('d', sqr_deviations)
	graph = ROOT.TGraph(histBins, x_arr, y_arr)

	graph.SetTitle(f"Norm. Sqr. Deviation per Bin ({sampleId});Reco. Jet Inv. Mass (GeV);Sqr. Norm. Deviation")
	graph.SetLineColor(ROOT.kRed)
	graph.SetLineWidth(2)

	c2 = ROOT.TCanvas("c2"+sampleId, f"Norm. Sqr. Deviation ({sampleId})", 1600, 1200)
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


	# Plot x_values vs deviations using ROOT
	y_dev_arr = array.array('d', deviations)
	graph_dev = ROOT.TGraph(histBins, x_arr, y_dev_arr)

	graph_dev.SetTitle(f"Norm. Deviation per Bin ({sampleId});Reco. Jet Inv. Mass (GeV);Norm. Deviation")
	graph_dev.SetLineColor(ROOT.kBlue+2)
	graph_dev.SetLineWidth(2)

	c3 = ROOT.TCanvas("c3"+sampleId, f"Norm. Deviation ({sampleId})", 1600, 1200)
	c3.SetLeftMargin(0.17)
	c3.SetBottomMargin(0.17)
	graph_dev.Draw("AL")

	# Add chi-squared values on the deviation plot as well
	latex_dev = ROOT.TLatex()
	latex_dev.SetNDC()
	latex_dev.SetTextFont(42)
	latex_dev.SetTextSize(0.04)
	latex_dev.SetTextColor(ROOT.kBlack)
	latex_dev.DrawLatex(0.22, 0.85, chi2_text)
	c3.Update()

	outputFile3 = f"{outputFolder + process.replace('/', '_')}_deviation_{branch}_{sampleId}.png"
	c3.SaveAs(outputFile3)
	print(f"Saved deviation plot to {outputFile3}")


	print("Chi-Squared = " + str(chi_sqr))
	print("Norm. Chi-Squared = " + str(chi_sqr / (histBins - 1)))
	print("\n")
