
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import math


# Analysis code name
analysisName = "sens_3prongs"


# Configuration
samples = [
    #"Mnutau_50p0MeV",
    #"Mnutau_100p0MeV",
    #"Mnutau_200p0MeV",
    "Mtau_m5p0MeV",
    "Mtau_m10p0MeV",
]

# Extract particle that is mass variated
particleName = samples[0].split("_")[0]

branches = [ "jet1_mass" ]

labels = {
    "Mtau_m5p0MeV":   "$\\Delta m_\\tau$ = -5 MeV",
    "Mtau_m10p0MeV":   "$\\Delta m_\\tau$ = -10 MeV / $2^2$",
    "Mnutau_50p0MeV":   "$m_\\nu$ = 50 MeV",
    "Mnutau_100p0MeV":  "$m_\\nu$ = 100 MeV / $2^4$",
    "Mnutau_200p0MeV":  "$m_\\nu$ = 200 MeV / $2^8$",
}

# Scaling power law with respect to mass
scalingPower = 1

# Scale the error bars by this amount
errorBarMultiplier = 1

rescaleFactors = {
    "Mnutau_50p0MeV":   1.0,
    "Mnutau_100p0MeV":  (1 / 2.0)**scalingPower,
    "Mnutau_200p0MeV":  (1 / 4.0)**scalingPower,
    "Mtau_m5p0MeV":  1,
    "Mtau_m10p0MeV":  (1 / 2.0)**scalingPower,
}


for branch in branches:

    # Prepare figure for all samples
    plt.figure(dpi=300)

    likelihoods = []
    chi_squares = []

    for sampleId in samples:

        print(f"Processing {branch} of sample {sampleId}...")

        # File paths
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
        #ratioHist.Divide(hist2)

        chi_sqr = 0
        logL = 0
        for i in range(1, hist1.GetNbinsX() + 1):

            Nexp = hist2.GetBinContent(i) / 100 # Rinormalize to MC, not data!
            Nobs = hist1.GetBinContent(i) / 100

            if Nobs != 0 and Nexp != 0:

                ratio = (Nobs / Nexp - 1) * rescaleFactors[sampleId]
                sigma_Nobs = math.sqrt(Nobs)
                sigma_Nexp = math.sqrt(Nexp)
                error = rescaleFactors[sampleId] * math.sqrt(
                    (sigma_Nobs / Nexp)**2 +
                    (Nobs * sigma_Nexp / Nexp**2)**2
                ) * errorBarMultiplier

                ratioHist.SetBinContent(i, ratio)
                ratioHist.SetBinError(i, error)

            else:
                ratioHist.SetBinContent(i, 0)
                ratioHist.SetBinError(i, 0)

            if Nexp != 0:
                chi_sqr += (Nexp - Nobs)**2 / Nexp
                logL += -2 * (Nobs * (math.log(Nexp / Nobs)) + (Nobs - Nexp))

        # Convert ROOT histogram to numpy arrays
        bin_edges = np.array([hist1.GetXaxis().GetBinLowEdge(i) for i in range(1, hist1.GetNbinsX() + 2)])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ratio_vals = np.array([ratioHist.GetBinContent(i) for i in range(1, ratioHist.GetNbinsX() + 1)])
        ratio_errs = np.array([ratioHist.GetBinError(i) for i in range(1, ratioHist.GetNbinsX() + 1)])

        plt.errorbar(
            bin_centers, ratio_vals, yerr=ratio_errs, fmt='o',
            label=labels[sampleId], 
            capsize=2, markersize=3, elinewidth=1
        )

        #plt.errorbar(
        #    bin_centers, ratio_vals, yerr=ratio_errs, fmt='o',
        #    ecolor='black', elinewidth=0.5, capsize=1,
        #    label=labels[sampleId],
        #)

        # Clean up
        file1.Close()
        file2.Close()

        print(f"{sampleId}: Log. Likelihood = \t{logL}")

        likelihoods.append(logL)
        chi_squares.append(chi_sqr)


    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.axvline(1.777, color='blue', linestyle='-', linewidth=1, label="Tau Mass")

    #plt.title(f"Mass Sensibility Comparison (Error Bars x25)", fontsize=16)
    plt.xlabel("Jet Invariant Mass (GeV)", fontsize=16)
    plt.ylabel("Rescaled Event Ratio", fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"hist_{particleName}_{analysisName}_compare_{branch}.png")
    plt.savefig(f"hist_{particleName}_{analysisName}_compare_{branch}.pdf")
    plt.close()

    # Plot likelihood vs neutrino mass
    #masses = [ 50, 100, 200 ]
    #plt.figure(figsize=(8, 6))
    #plt.plot(masses, likelihoods, marker='o', linestyle='-', color='purple')
    #plt.title("Likelihood vs Neutrino Mass", fontsize=16)
    #plt.xlabel("Neutrino Mass (MeV)", fontsize=14)
    #plt.ylabel("Log. Likelihood", fontsize=14)
    #plt.grid(True, linestyle=':', alpha=0.5)
    #plt.tight_layout()
    #plt.savefig(f"logL_vs_mass_{branch}.png", dpi=300)
    #plt.close()

