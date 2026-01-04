
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit


# Configuration

# Code name for the analysis
analysisName = "sens_3prongs"

# Coupling constant between tau neutrino and HNL
mixingValues = [ 2**(-n) for n in range(0, 14) ]

samples = [
    "Mnutau_10p0MeV",
    "Mnutau_50p0MeV",
    "Mnutau_100p0MeV",
    "Mnutau_200p0MeV",
]

sampleMasses = [
    10,
    50,
    100,
    200
]

branch = "jet1_mass"

# How high to cut on the likelihood using the graphical method
likelihoodCut = 2.71


# For each value of mixing constant, estimate the E.L.
exclusionLimits = []
for mixing in mixingValues:

    print(f"Processing mixing {mixing}")

    likelihoods = []
    effectiveMasses = []
    
    # Compute the likelihoods for each mass
    for i in range(len(samples)):

        sample = samples[i]

        # File paths
        filePath1 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_ecm91.root"
        filePath2 = f"outputs/histmaker/Ztautau_{analysisName}/p8_ee_Ztautau_{sample}_ecm91.root"
        histName = branch

        # Load histograms from ROOT files
        file1 = ROOT.TFile.Open(filePath1)
        file2 = ROOT.TFile.Open(filePath2)

        histCentral = file1.Get(histName)
        histVariation = file2.Get(histName)

        # Compute the log likelihood
        logL = 0
        for j in range(0, histCentral.GetNbinsX() + 1):

            N_m0 = histCentral.GetBinContent(j)
            N_mvar = histVariation.GetBinContent(j)

            Nobs = N_m0

            # Simple mixing model of a single HNL, using effective mass
            # and predominant mixing with tau neutrino
            Nexp = (1 - mixing) * N_m0 + mixing * N_mvar

            if Nobs != 0 and Nexp != 0:
                logL += -2 * (Nobs * (math.log(Nexp / Nobs)) + (Nobs - Nexp))


        print(f"{branch}\t{sample}\tLog Likelihood:\t{logL}")
        likelihoods.append(logL)
        effectiveMasses.append(sampleMasses[i])

        # Clean up
        file1.Close()
        file2.Close()


    # Fit the likelihood with the model
    print("Fitting likelihood with quartic model...")
    effectiveMasses.append(0)
    likelihoods.append(0)
    
    x_vals = np.array(effectiveMasses, dtype=float)
    y_vals = np.array(likelihoods, dtype=float)
    graph = ROOT.TGraph(len(x_vals), x_vals, y_vals)
    model = ROOT.TF1("model", "[0]*x*x*x*x + [1]*x*x", min(x_vals), max(x_vals) + 1)
    fitResult = graph.Fit(model, "QRS")
    fitErrors = [ fitResult.ParError(0), fitResult.ParError(1) ]
    
    fitParams = fitResult.GetParams()
    a, b = fitParams[0], fitParams[1]
    coeffs = [a, 0, b, 0, -likelihoodCut]
    print(f"Fit parameters: {fitParams[0]}, {fitParams[1]}")

    canvas = ROOT.TCanvas("c"+str(i)+str(j)+str(mixing), "Quartic Fit", 1600, 1200)
    canvas.SetLeftMargin(0.15)
    graph.SetTitle(
        f"HNL Likelihood Model (#alpha = {mixing});HNL Mass (MeV);Log. Likelihood"
    )
    graph.SetMarkerStyle(20)
    graph.Draw("AP")

    # Draw fit parameters on canvas
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.2, 0.85, "y = a x^4 + b x^2")
    latex.DrawLatex(0.2, 0.80, f"a = {fitParams[0]:.3f} #pm {fitErrors[0]:.3f}")
    latex.DrawLatex(0.2, 0.75, f"b = {fitParams[1]:.3f} #pm {fitErrors[1]:.3f}")

    canvas.SaveAs(f"fit_HNL_{mixing}.png")

    # Cut the likelihood to estimate the Exclusion Limit
    print(f"Estimating 95% Exclusion Limit...")
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r)]
    exclusionLimit = max(abs(r) for r in real_roots) if real_roots else float('nan')
    exclusionLimits.append(exclusionLimit)
    print("")


plt.figure(dpi=300)

# Fit the exclusion limit with respect to the branching ratio
x = mixingValues
y = exclusionLimits

def model(x, a):
    return a / x


#fitParams, fitCovar = curve_fit(model, x, y, maxfev=10000)
#inverseParam = fitParams[0]
#x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
#y_fit = model(x_fit, inverseParam)
#plt.plot(x_fit, y_fit, 'r--', label=f'Fit: $y = {inverseParam:.2e} / x')
#plt.legend()
#print(f"Power law fit: y = {inverseParam:.2e} / x")


# Plot the Exclusion Level for varying mixing coefficient
plt.plot(mixingValues, exclusionLimits, marker='o', linestyle='-')
#plt.text(0.5, 0.8, "$N_{events} = (1 - \\alpha) N_{\\nu} + \\alpha N_{HNL}$", transform=plt.gca().transAxes, fontsize=12)

plt.xlabel('Branching Ratio $\\alpha$', fontsize=15)
plt.ylabel(f'95% Exclusion Limit on HNL Mass (MeV)', fontsize=15)

plt.yscale("log")
plt.xscale("log")

plt.ylim(1e-02, 5e+02)

#plt.title(f'Heavy Neutral Lepton 95% E.L.')
plt.tight_layout()
plt.grid(True)

plt.savefig("el_HNL_mixing.png")
plt.savefig("el_HNL_mixing.pdf")
