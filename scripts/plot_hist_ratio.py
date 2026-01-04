
import ROOT
import numpy as np
import matplotlib.pyplot as plt


# Configuration
samples = [
    "Mtau_m5p0MeV",
    "Mtau_p5p0MeV",
    "Mtau_m10p0MeV",
    "Mtau_p10p0MeV",
    #"Mnutau_0p1MeV",
    #"Mnutau_1p0MeV",
    #"Mnutau_10p0MeV",
    #"Mnutau_100p0MeV",
]

branches = [
    "jet1_mass",
    #"jets_reco_mass"
]


for sampleId in samples:

    for branch in branches:

        print(f"Processing {branch} of sample {sampleId}...")

        # File paths
        analysisName = "sens_5prongs"
        filePath1 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_ecm91.root"
        filePath2 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_{sampleId}_ecm91.root"
        hist_name = branch

        # Open ROOT files
        file1 = ROOT.TFile.Open(filePath1)
        file2 = ROOT.TFile.Open(filePath2)

        # Get histograms
        hist1 = file1.Get(hist_name)
        hist2 = file2.Get(hist_name)

        # Clone hist1 for ratio and divide
        ratioHist = hist1.Clone("ratio_hist"+sampleId+branch)
        ratioHist.Divide(hist2)

        chi_sqr = 0
        for i in range(1, hist1.GetNbinsX() + 1):

            Nexp = hist1.GetBinContent(i)
            Nobs = hist2.GetBinContent(i)

            if Nexp != 0:
                chi_sqr += (Nexp - Nobs)**2 / Nexp

        # Convert ROOT histogram to numpy arrays
        bin_edges = np.array([hist1.GetXaxis().GetBinLowEdge(i) for i in range(1, hist1.GetNbinsX() + 2)])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ratio_vals = np.array([ratioHist.GetBinContent(i) for i in range(1, ratioHist.GetNbinsX() + 1)])
        ratio_errs = np.array([ratioHist.GetBinError(i) for i in range(1, ratioHist.GetNbinsX() + 1)])

        plt.figure(figsize=(8, 6))
        plt.errorbar(bin_centers, ratio_vals, yerr=ratio_errs, fmt='o', ecolor='black', elinewidth=0.5, capsize=1, label='Event Ratio')

        plt.axhline(1, color='red', linestyle='--', linewidth=1, label='Ratio = 1')

        plt.title(f"Mass Sensibility ({sampleId.replace('_', ' ')})", fontsize=16)
        plt.xlabel("Jet Invariant Mass (GeV)", fontsize=14)
        plt.ylabel("Event Ratio", fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.5)
        #plt.legend(fontsize=12)

        # Annotate chi-squared values
        plt.text(
            0.05, 0.95, f"$\chi^2$ = {chi_sqr:.2f}",
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plt.tight_layout()
        plt.savefig(f"hist_ratio_{sampleId}_{branch}_5prongs.png", dpi=300)
        plt.close()

        # Clean up
        file1.Close()
        file2.Close()
        