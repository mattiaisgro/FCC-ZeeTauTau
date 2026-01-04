
import ROOT
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq


# Configuration

# Analysis code name
analysisName = "Mtau"

# Sample code names
sampleNames = [
    'p8_ee_Ztautau_ecm91',
    'p8_ee_Ztautau_Mtau_m5p0MeV_ecm91',
    'p8_ee_Ztautau_Mtau_p5p0MeV_ecm91',
    'p8_ee_Ztautau_Mtau_m10p0MeV_ecm91',
    'p8_ee_Ztautau_Mtau_p10p0MeV_ecm91',
]
# NOTE: All histograms should have the same binning and number of bins.

# Mass variations in MeV
sampleMasses = [
    0,
    -5, +5,
    -10, +10
]

# Input folder containing the histograms
inputFolder = "./outputs/histmaker/Ztautau_sens_3prongs/"

# The branch to analyze
branchName = 'jet1_mass'

# Total integrated luminosity
luminosityTotal = 205 # /ab
luminosityFactors = [ 2**(i-40) for i in range(0, 42) ]
luminosityValues = [ luminosityTotal * a for a in luminosityFactors ]


# Mass model function
def fitFunc(mass, a, b):
    return a + b * mass


# Load histograms
print("Loading histograms...")
histograms = []
for sample in sampleNames:

    f = ROOT.TFile.Open(inputFolder + sample + ".root")
    if not f or f.IsZombie():
        raise RuntimeError(f"File '{inputFolder + sample + '.root'}' could not be opened.")
    
    hist = f.Get(branchName)
    if not hist or hist.InheritsFrom("TH1") == 0:
        f.Close()
        raise RuntimeError(f"Histogram '{branchName}' not found in file '{inputFolder + sample + '.root'}'")

    hist.SetDirectory(0)
    histograms.append(hist)
    f.Close()


# Prepare data for each bin
nBins = histograms[0].GetNbinsX()
binCenters = [histograms[0].GetBinCenter(i+1) for i in range(nBins)]


# Results of the fits for each bin
fitResults = {}

# Compute the fit
print("Fitting histograms...")
for binIndex in range(nBins):

    y = []
    yErr = []

    # Collect bin contents and errors for each sample
    for hist in histograms:
        y.append(hist.GetBinContent(binIndex+1))
        yErr.append(hist.GetBinError(binIndex+1))

    # Normalize y and yErr with respect to the maximum y value
    y_max = max(y) if max(y) != 0 else 1.0
    y = [val / y_max for val in y]
    yErr = [err / y_max for err in yErr]

    # Fit number of events to mass variations
    try:
        params, covariance = curve_fit(fitFunc, sampleMasses, y, sigma=yErr, absolute_sigma=True)
    except Exception as e:
        params = [np.nan, np.nan, np.nan]
        covariance = np.full((2, 2), np.nan)

    # Rescale parameters to match original (unnormalized) y values
    params_rescaled = [p * y_max for p in params]
    covariance_rescaled = covariance * (y_max ** 2)

    fitResults[binIndex] = {'params': params_rescaled, 'cov': covariance_rescaled}

    # Plot the fit results and error bars
    if binIndex % 20 == 0:
        plt.figure(dpi=300)
        plt.errorbar(sampleMasses, y, yerr=yErr, fmt='o', label='Data')
        mass_fit = np.linspace(
            min(sampleMasses) - abs(max(sampleMasses)) * 0.1,
            max(sampleMasses) + abs(max(sampleMasses)) * 0.1,
            100
        )
        fit_vals = fitFunc(mass_fit, *params)
        plt.plot(mass_fit, fit_vals, '-', label='Fit')
        plt.xlabel('Mass Variation (MeV)')
        plt.ylabel('Events in Bin {}'.format(binIndex))
        plt.title(f'Fit for Bin {binIndex}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fit_bin_{binIndex}_{analysisName}.png')
        plt.close()


# Example: function to compute number of events for a given mass and bin
def predictEvents(binIndex, mass):
    params = fitResults[binIndex]['params']
    return fitFunc(mass, *params)


# Compute the log. likelihood for a given mass value/variation
def predictLikelihood(mass):

    likelihood = 0.0
    for binIndex in range(nBins):
        
        Nexp = predictEvents(binIndex, mass)
        Nobs = histograms[0].GetBinContent(binIndex+1)

        # Compute the log likelihood for this bin
        likelihood += -2 * (Nobs * math.log(Nexp / Nobs) + (Nobs - Nexp))
    
    return likelihood


# Plot the log. likelihood for a range of mass values
print("Plotting log likelihood model...")

masses = np.linspace(min(sampleMasses), max(sampleMasses), 100)
likelihoods = [predictLikelihood(m) for m in masses]

# Normalize the likelihood to zero
minLikelihood = predictLikelihood(0)
for i in range(len(likelihoods)):
    likelihoods[i] = likelihoods[i] - minLikelihood

# Fit a model to the predicted parabola
likelihoodModel = ROOT.TF1("likelihoodModel", "[0]*x*x + [1]", min(masses), max(masses))
likelihoodModel.SetParameter(1, 0.0)
x_vals = np.array(masses, dtype=float)
y_vals = np.array(likelihoods, dtype=float)
graph = ROOT.TGraph(100, x_vals, y_vals)
fitResult = graph.Fit(likelihoodModel, "RS")
fitParams = fitResult.GetParams()

# Save a plot of the likelihood function
plt.figure(dpi=300)
plt.plot(masses, likelihoods, label='Log Likelihood', color='blue')
plt.xlabel('Mass Variation (MeV)')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood vs Mass Variation')
#plt.axhline(0, color='red', linestyle='--', label='Zero Line')
plt.legend()
#plt.grid()
plt.show()
plt.savefig(f'likelihood_model_{analysisName}.png')
plt.savefig(f'likelihood_model_{analysisName}.pdf')


sigmas = []

for i in range(len(luminosityValues)):
    # Find zeroes of the likelihood
    coeffs = [fitParams[0] * luminosityFactors[i], 0, fitParams[1] * luminosityFactors[i] - 1]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r)]
    sigma = max(abs(r) for r in real_roots) if real_roots else float('nan')
    sigmas.append(sigma)
    print(f"Luminosity: {luminosityValues[i]:.2f} ab^-1, Mass Uncertainty: {sigma}")


# Extrapolate values following inverse law from sigmas[0]
extrValues = []
scalarFactor = sigmas[0] * (luminosityTotal * luminosityFactors[0])**0.5

for luminosityFactor in luminosityFactors:
    extrValues.append(scalarFactor / (luminosityFactor * luminosityTotal)**0.5)

# Plot sigma vs luminosity
plt.figure(dpi=300)
plt.plot(luminosityValues, sigmas, marker='o')
plt.plot(luminosityValues, extrValues, linestyle='--')
plt.xlabel('Luminosity (ab$^{-1}$)')
plt.ylabel('Mass Uncertainty (sigma, MeV)')
plt.title('Mass Uncertainty vs Luminosity')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'sigma_vs_luminosity_{analysisName}.png')
plt.show()

print(f"Mass Uncertainty at the FCC-ee Z pole: {extrValues[-2]}")
