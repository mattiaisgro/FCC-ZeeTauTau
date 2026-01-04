
import ROOT
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math


# Convert a mass variation to a sample code name (e.g. 50 MeV -> p50p0MeV)
def massToStr(mass, use_sign=True):

    res = ""
    if use_sign:
        if mass > 0:
            res += "p"
        else:
            res += "m"

    res += f"{abs(mass):.1f}MeV"
    res = res.replace(".", "p")

    return res


# Compute the log-likelihood between two histograms
def getHistLogLikelihood(tauHist, neutrinoHist):

    # Compute the log-likelihood
    logLikelihood = 0
    for i in range(1, tauHist.GetNbinsX() + 1):

        tauContent = tauHist.GetBinContent(i)
        neutrinoContent = neutrinoHist.GetBinContent(i)

        if tauContent > 0 and neutrinoContent > 0:

            Nobs = tauContent
            Nexp = neutrinoContent

            if Nobs != 0 and Nexp != 0:
                logLikelihood += -2 * (Nobs * (math.log(Nexp / Nobs)) + (Nobs - Nexp))

    return logLikelihood


# Total integrated luminosity
luminosityTotal = 205 # /ab

# Cross section in the sample
crossSection = 1476.58 # pb

# Cut on the likelihood likelihoodModel for 95% Exclusion Limit
likelihoodCut = 2.71

# Tau Mass variations
tauMassVariations = [ 0, +5, +10]

# Neutrino mass variations
neutrinoMassVariations = [ 0, 10, 50, 100, 200 ]

# Independent measure of tau mass uncertainty
tauStdev = 0.15 # MeV

# Branch to consider in fit
branch = "jet1_mass"

# Folder where the ROOT files are
inputFolder = "./outputs/histmaker/Ztautau_sens_3prongs/"


# List of tau mass-varied samples
tauSamples = [
    f"p8_ee_Ztautau_Mtau_{massToStr(mass)}_ecm91"
    if mass != 0 else "p8_ee_Ztautau_ecm91"
    for mass in tauMassVariations
]

# List of tau neutrino mass-varied samples
neutrinoSamples = [
    f"p8_ee_Ztautau_Mnutau_{massToStr(mass, False)}_ecm91"
    if mass != 0 else "p8_ee_Ztautau_ecm91"
    for mass in neutrinoMassVariations
]


print("Tau samples:")
for sample in tauSamples:
    print("\t" + sample)
print()

print("Neutrino samples:")
for sample in neutrinoSamples:
    print("\t" + sample)
print()


# Compute the exclusion limits for varied tau masses
exclusionLimits = []
for tauIndex in range(len(tauSamples)):

    print(f"Processing tau mass variation {tauMassVariations[tauIndex]} MeV")
    tauFile = ROOT.TFile.Open(inputFolder + tauSamples[tauIndex] + ".root")
    tauHist = tauFile.Get(branch)

    likelihoods = []
    for neutrinoIndex in range(len(neutrinoSamples)):

        print(f"\tProcessing neutrino mass variation {neutrinoMassVariations[neutrinoIndex]} MeV")
        neutrinoFile = ROOT.TFile.Open(inputFolder + neutrinoSamples[neutrinoIndex] + ".root")
        neutrinoHist = neutrinoFile.Get(branch)

        # Check if histogram is valid and has entries
        if not neutrinoHist or neutrinoHist.GetEntries() == 0:
            print(f"\t\tWarning: Histogram for {neutrinoSamples[neutrinoIndex]} is empty or missing.")
            likelihood = float('nan')
        else:
            # Compute the likelihood between expected and observed models
            likelihood = getHistLogLikelihood(tauHist, neutrinoHist)

        likelihoods.append(likelihood)


    # Fit likelihoods to the quartic model
    xValues = np.array(neutrinoMassVariations, dtype=float)
    yValues = np.array(likelihoods, dtype=float)

    likelihoodGraph = ROOT.TGraph(len(xValues), xValues, yValues)
    likelihoodModel = ROOT.TF1(
        f"likelihoodModel_{tauMassVariations[tauIndex]}_{neutrinoMassVariations[neutrinoIndex]}",
        "[0]*x*x*x*x + [1]*x*x", min(xValues), max(xValues) + 1
    )

    print("\tFitting likelihood to quartic model...")
    fitResult = likelihoodGraph.Fit(likelihoodModel, "RS")
    fitParams = fitResult.GetParams()
    fitErrors = [ fitResult.ParError(0), fitResult.ParError(1) ]


    # Plot the likelihood model
    fig_fit, ax_fit = plt.subplots(figsize=(6, 6), dpi=300)
    ax_fit.set_xlabel("Neutrino Mass Variation (MeV)", fontsize=15)
    ax_fit.set_ylabel("Log Likelihood", fontsize=15)
    xFit = np.linspace(min(neutrinoMassVariations), max(neutrinoMassVariations), 100)
    yFit = fitParams[0] * xFit**4 + fitParams[1] * xFit**2
    ax_fit.plot(xFit, yFit, '-', color='red')
    ax_fit.scatter(neutrinoMassVariations, likelihoods, marker='o',  color='b', label='Log Likelihood')


    # Compute the exclusion limit
    coeffs = [fitParams[0], 0, fitParams[1], 0, -likelihoodCut]
    roots = np.roots(coeffs)
    realRoots = [r.real for r in roots if np.isreal(r)]

    exclusionLimit = max(abs(r) for r in realRoots) if realRoots else float('nan')
    exclusionLimits.append(exclusionLimit)

    # Print the exclusion limit on top of the plot
    ax_fit.text(
        0.1, 0.8,
        f"95% EL = {exclusionLimit * 1000:.1f} keV",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        transform=ax_fit.transAxes
    )

    #ax_fit.legend()
    plt.tight_layout()
    plt.savefig(f"./Ztautau_likelihood_{branch}_Mtau_{massToStr(tauMassVariations[tauIndex])}.png")
    plt.close()

    print("")


print(tauMassVariations)
print(exclusionLimits)


# Fit the exclusion limit vs. the tau mass variations
print("Computing exclusion limit model...")
xValues = np.array(tauMassVariations, dtype=float)
yValues = np.array(exclusionLimits, dtype=float)

def elModel(x, a, b):
    return a*x + b

fitParams, fitCov = curve_fit(elModel, xValues, yValues)
fitErrors = np.sqrt(np.diag(fitCov))

# Plot the exclusion limit model
fig_fit, ax_fit = plt.subplots(figsize=(6, 6), dpi=300)
ax_fit.set_xlabel("Tau Mass Variation (MeV)", fontsize=15)
ax_fit.set_ylabel("95%% Exclusion Limit (MeV)", fontsize=15)
xFit = np.linspace(min(tauMassVariations) - 2, max(tauMassVariations) + 2, 100)
yFit = elModel(xFit, fitParams[0], fitParams[1])
ax_fit.plot(xFit, yFit, '-', color='red')
ax_fit.scatter(tauMassVariations, exclusionLimits, marker='o', linestyle='-', color='b', label="Asimov Fit")
ax_fit.axvline(tauStdev, color='green', linestyle='--', label="Tau Uncertainty")
ax_fit.axvline(-tauStdev, color='green', linestyle='--')


# Estimate variation for the tau neutrino
centralValue = elModel(0, fitParams[0], fitParams[1])
upperCut = abs(elModel(tauStdev, fitParams[0], fitParams[1]) - centralValue)
lowerCut = abs(elModel(-tauStdev, fitParams[0], fitParams[1]) - centralValue)

ax_fit.text(
    0.7, 0.6,
    f"$\\Delta$EL = {max(upperCut, lowerCut) * 1000:.2f} keV",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
    transform=ax_fit.transAxes
)

ax_fit.legend()
plt.tight_layout()
plt.savefig(f"./Ztautau_exclusion_{branch}.png")
plt.close()
