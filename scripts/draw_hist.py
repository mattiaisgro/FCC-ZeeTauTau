
import ROOT
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import math


# Configuration
samples = [
    "p8_ee_Ztautau_ecm91",
    #"p8_ee_Ztautau_Mtau_m5p0MeV_ecm91",
    #"p8_ee_Ztautau_Mtau_p5p0MeV_ecm91",
    #"p8_ee_Ztautau_Mtau_m10p0MeV_ecm91",
    "p8_ee_Ztautau_Mtau_p10p0MeV_ecm91",
    #"p8_ee_Ztautau_Mnutau_100p0MeV_ecm91",
    "p8_ee_Ztautau_Mnutau_200p0MeV_ecm91",
]

sampleNames = [
    "$Z \\rightarrow \\tau \\tau$ (central)",
    #"$Z \\rightarrow \\tau \\tau$ ($\Delta m_{\\tau}$ = -5 MeV)",
    #"$Z \\rightarrow \\tau \\tau$ ($\Delta m_{\\tau}$ = +5 MeV)",
    #"$Z \\rightarrow \\tau \\tau$ ($\Delta m_{\\tau}$ = -10 MeV)",
    "$Z \\rightarrow \\tau \\tau$ ($\Delta m_{\\tau}$ = +10 MeV)",
    #"$Z \\rightarrow \\tau \\tau$ $(m_{\\nu}$ = 100 MeV)",
    "$Z \\rightarrow \\tau \\tau$ $(m_{\\nu}$ = 200 MeV)",
]

inputFolder = "./outputs/histmaker/Ztautau_sens_3prongs/"

branches = [
    #"nlep", "lep_p", "lep_theta", "lep_phi",
    "jet1_mass",
    #"jet1_p", "jet2_mass",
    #"jets_total_mass", "nprongs",
    #"jet1_total_charge", "jet2_total_charge",
    #"jet1_theta", "jet2_theta",
    #"jet1_phi", "jet2_phi",
]

outputFolder = "./"
os.makedirs(outputFolder, exist_ok=True)


# Load an histogram from a ROOT file
def load_histogram(file_path, branch):

    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {file_path} (Zombie file or not found)")
        return None

    hist = f.Get(branch)
    if not hist:
        print(f"Could not find histogram {branch} in {file_path}")
        f.Close()
        return None

    hist.SetDirectory(0)
    f.Close()
    return hist


# Plot histograms for each sample
def plot_histograms(hist_list, labels, branch, output_path):

    #colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    plt.figure(figsize=(7, 5), dpi=300)

    for i, hist in enumerate(hist_list):

        if not hist:
            continue

        # Convert ROOT histogram to numpy arrays
        n_bins = hist.GetNbinsX()
        bin_edges = np.array([hist.GetBinLowEdge(j+1) for j in range(n_bins)] + [hist.GetBinLowEdge(n_bins+1)])
        #totalEvents = hist.Integral()
        values = np.array([hist.GetBinContent(j+1) for j in range(n_bins)])

        # Plot line segments
        #plt.plot(bin_edges[:-1], values, linewidth=2, label=sampleNames[i])

        # Plot steps histogram
        #plt.steps(bin_edges[:-1], values, linewidth=2, label=sampleNames[i])

        # Plot smooth line
        X_smooth = np.linspace(bin_edges[0], bin_edges[-1], 300)
        spline = make_interp_spline(bin_edges[:-1], values, k=3)
        Y_smooth = spline(X_smooth)
        plt.plot(X_smooth, Y_smooth, label=sampleNames[i])

        # Add background color
        #plt.fill_between(bin_edges[:-1], values, step='post', alpha=0.3, color='lightblue')

    #plt.title(branch, fontsize=16)
    plt.xlabel("Invariant Jet Mass (GeV)", fontsize=15)
    plt.ylabel("Entries", fontsize=15)
    #plt.yscale("log")

    #plt.axvline(x=1.0, color="red", linestyle='--', linewidth=1.5, label="Fit Region")
    #plt.axvline(x=2.0, color="red", linestyle='--', linewidth=1.5)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path + ".png")
    plt.savefig(output_path + ".pdf")
    plt.close()


for branch in branches:

    print(f"Processing histogram: {branch}")

    hists = []
    for sample in samples:

        file_path = os.path.join(inputFolder, f"{sample}.root")
        hist = load_histogram(file_path, branch)
        hists.append(hist)

    output_path = os.path.join(outputFolder, f"hist_Ztautau_{branch}")
    plot_histograms(hists, samples, branch, output_path)
    print(f"Saved plot to {output_path}")
