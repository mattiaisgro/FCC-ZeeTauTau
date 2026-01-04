
import ROOT
from matplotlib import pyplot as plt
import numpy as np
import math


# New Reco - Tau
#likelihoods = [ 0, 9877898.700569278, 9824560.318010258 ]
#name = "Mtau_reco_new"

# Old Reco - Tau
#likelihoods = [ 0, 8067420.229865246, 8437265.4155649 ]
#name = "Mtau_reco_old"


# New Reco - Neutrino
#likelihoods = [ 0, 1373227.4177301996, 175536275.8599682 ]
#name = "Mnutau_reco_new"

# Old Reco - Neutrino
#likelihoods = [ 0, 1215918.8710811764, 132723063.92532705 ]
#name = "Mnutau_reco_old"


# New Reco - Tau 5-Prongs
#likelihoods = [ 0, 1289944.2257032103, 1180927.5480101344 ]
#name = "Mtau_reco_new_5prongs"

# Old Reco - Tau 5-Prongs
#likelihoods = [ 0, 524164.741865024, 565033.3910302944 ]
#name = "Mtau_reco_old_5prongs"


# New Reco - Neutrino 5-Prongs
#likelihoods = [ 0, 702393.8104992469, 38470106.17886722 ]
#name = "Mnutau_reco_new_5prongs"

# Old Reco - Neutrino 5-Prongs
#likelihoods = [ 0, 584535.9181686855, 746360.4060121874 ]
#name = "Mnutau_reco_old_5prongs"


# New Reco - Neutrino 3-Prongs with 4 mass values
likelihoods = [ 0, 654892, 13095162, 174683308, 2321858280 ]
name = "Mnutau_3prongs"

# New Reco - Tau 3-Prongs with 4 (2) mass values
#likelihoods = [
#    0,
#    #2720944, 2881688,
#    9175024, 9198756
#]
#name = "Mtau_3prongs"


# New Reco - Neutrino 5-Prongs with 4 mass values
#likelihoods = [ 0, 699280.7, 4816807.3, 38650441, 255798053.7 ]
#name = "Mnutau_5prongs"

# New Reco - Tau 5-Prongs with 4 (2) mass values
#likelihoods = [
#    0,
#    #745569, 714865,
#    1280301, 1175640
#]
#name = "Mtau_5prongs"

# New Reco - Tau combined (3-prongs + 5-prongs)
#likelihoods = [
#    0, 1280301 + 9175024, 1175640 + 9198756
#]
#name = "Mtau_combined"

# New Reco - Neutrino combined (3-prongs + 5-prongs)
#likelihoods = [ 0, 654892 + 699280.7, 4816807.3 + 13095162, 38650441 + 174683308, 255798053.7 + 2321858280 ]
#name = "Mnutau_combined"


displayName = name.replace("_", " ")

#fit_type = "quadratic"
fit_type = "quartic"

# Cut on the likelihood curve (1 = 1*sigma, 4 = 2*sigma, 2.71 = 2-sigma 1-sided ...)
#likelihoodCut = 1
likelihoodCut = 2.71

# Number of events in MC sample
Nmc = 1e+8

# Cross section in the sample
crossSection = 1476.58 # pb

# Luminosity such that N(MC) = N(Data)
Leq = (Nmc / crossSection) * 1e-06


# X values
#massVariations = [ 0, -10, +10 ]  # Tau Mass variations
massVariations = [ 0, 10, 50, 100, 200 ] # Neutrino mass variations

# Scaling power for extrapolation (0.5 for quadratic, 0.25 for quartic)
scalingPower = 0.25

# Total integrated luminosity
luminosityTotal = 205 # /ab

# Mass of the tau lepton
tauMass = 1776.93 # Mev

# Luminosity reduction factor
luminosityFactors = [ 2**(i-45) for i in range(0, 47) ]
print("Luminosity Factors = " + str(luminosityFactors))


lumValues = []
sigmaValues = []

for i in range(0, len(luminosityFactors)):

    luminosityFactor = luminosityFactors[i]
    print(f"Processing luminosity {luminosityTotal * luminosityFactor}")

    x_vals = np.array(massVariations, dtype=float)
    y_vals = np.array(likelihoods, dtype=float) * luminosityFactor

    # Create a TGraph
    n_points = len(x_vals)
    graph = ROOT.TGraph(n_points, x_vals, y_vals)

    # Fit with a quadratic or quartic function
    if fit_type == "quadratic":

        # Define custom quadratic function with b=0 (derivative is zero in the origin)
        func = ROOT.TF1("func", "[0]*x*x + [1]", min(x_vals), max(x_vals))
        func.SetParameter(1, 0.0)
        fit_result = graph.Fit(func, "RS")

    elif fit_type == "quartic":
        func = ROOT.TF1("func", "[0]*x*x*x*x", min(x_vals), max(x_vals) + 1)
        fit_result = graph.Fit(func, "RS")
    else:
        raise ValueError("fit_type must be 'quadratic' or 'quartic'")

    fit_params = fit_result.GetParams()

    if fit_type == "quadratic":
        fit_errors = [fit_result.ParError(i) for i in range(2)]
    elif fit_type == "quartic":
        fit_errors = [fit_result.ParError(i) for i in range(1)]

    #print(f"Fit parameters: {fit_params[0]}, {fit_params[1]}")
    print(f"Fit parameters: {fit_params[0]}")

    # Optional: Draw the graph and fit using Matplotlib
    fig_fit, ax_fit = plt.subplots(figsize=(6, 6), dpi=300)
    ax_fit.set_xlabel("Mass Variation (MeV)", fontsize=15)
    ax_fit.set_ylabel("Log Likelihood", fontsize=15)
    

    # Plot the fit curve
    x_fit = np.linspace(min(x_vals), max(x_vals), 200)
    if fit_type == "quadratic":
        a, b = fit_params[0], fit_params[1]
        y_fit = a * x_fit**2 + b
    elif fit_type == "quartic":
        #a, b = fit_params[0], fit_params[1]
        a = fit_params[0]
        y_fit = a * x_fit**4
    else:
        raise ValueError("fit_type must be 'quadratic' or 'quartic'")

    ax_fit.plot(x_fit, y_fit, '-', color='red')
    ax_fit.plot(x_vals, y_vals, 'o', color='blue')

    # Compute intersections between the fit and y=likelihoodCut
    if fit_type == "quadratic":
        coeffs = [a, 0, b - likelihoodCut]
    elif fit_type == "quartic":
        coeffs = [a, 0, 0, 0, -likelihoodCut]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.isreal(r)]

    print(f"Roots: {real_roots}")

    sigma = max(abs(r) for r in real_roots) if real_roots else float('nan')

    # Annotate fit parameters and sigma
    if fit_type == "quadratic":
        param_text = f"$\\log L = a m^2 + b$\na = {a:.4g}\nb = {b:.4g}"
    else:
        #param_text = f"$\\log L = a m^4 + b m^2$\na = {a:.4g}\nb = {b:.4g}"
        param_text = f"$\\log L = a m^4$\na = {a:.4g}"

    ax_fit.text(
        0.1, 0.95, param_text, transform=ax_fit.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    #ax_fit.legend()
    plt.tight_layout()

    # Save the plot for the second-to-last luminosity factor
    if i == (len(luminosityFactors) - 2):

        #ax_fit.plot([-5, +5], [2720944, 2881688], 'o', color='green')

        fig_fit.savefig(f"fit_likelihood_{name}_mass.png")

    plt.close(fig_fit)

    print(f"\nluminosity = {luminosityFactor * luminosityTotal:.2f}, sigma = {sigma:.2f}\n")
    lumValues.append(luminosityFactor * luminosityTotal)
    sigmaValues.append(sigma)


# Extrapolate values following inverse law from sigmaValues[0]
extrValues = []
scalarFactor = sigmaValues[0] * (luminosityTotal * luminosityFactors[0])**scalingPower

for luminosityFactor in luminosityFactors:
    extrValues.append(scalarFactor / (luminosityFactor * luminosityTotal)**scalingPower)


# Extrapolate another line when the true scaling kicks in
extrValues2 = []
extrIndex2 = 24

if fit_type == "quartic":

    referenceLuminosity = luminosityTotal * luminosityFactors[extrIndex2]
    referenceSigma = sigmaValues[extrIndex2]
    scalarFactor2 = referenceSigma * (referenceLuminosity)**0.5

    for luminosityFactor in luminosityFactors:
        extrValues2.append(scalarFactor2 / (luminosityFactor * luminosityTotal)**0.5)

# Standard deviation for the integrated luminosity
extrStdev = sigmaValues[-2]


# Convert lumValues and sigmaValues to ROOT arrays
lumArray = np.array(lumValues, dtype=float)
sigmaArray = np.array(sigmaValues, dtype=float)
extrArray = np.array(extrValues, dtype=float)

if fit_type == "quartic":
    extrArray2 = np.array(extrValues2, dtype=float)


# Convert to ppm for tau mass
if fit_type == "quadratic":
    sigmaArray = sigmaArray / tauMass * 1e+06 
    extrArray = extrArray / tauMass * 1e+06


# Create Matplotlib figure and axis
fig, ax = plt.subplots(dpi=300)
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_ybound(1, 1e+05)

# Plot fit graph
ax.plot(lumArray, sigmaArray, marker='o', color='blue', linewidth=2, label='Asimov fit', markersize=5)

# Plot extrapolation graph
ax.plot(lumArray, extrArray, linestyle='--', color='red', linewidth=2, label='Quartic Stat. Scaling')

# Plot the new extrapolation line
if fit_type == "quartic":
    ax.plot(lumArray, extrArray2, linestyle='--', color='green', linewidth=2, label='Quadratic Stat. Scaling')

# Draw vertical red lines at x = Leq/100 and x = 205
#ax.axvline(Leq / 100, color='red', linewidth=2, label='$N_{MC} = 100 N_{data}$')
#ax.axvline(Leq / 10, color='orange', linewidth=2, label='$N_{MC} = 10 N_{data}$')
#ax.axvline(1e-1, color='orange', linewidth=2, label='LEP Z pole run')
ax.axvline(205, color='green', linewidth=2, label='FCC-ee Z pole run')

# Set axis labels and title
#ax.set_title(f"Mass Stat. Uncertainty vs Luminosity ({displayName})")
ax.set_xlabel("Luminosity ($ab^{-1}$)", fontsize=15)

#ax.set_ylabel(f"Statistical uncertainty on $\\tau$ mass (ppm)", fontsize=15)
ax.set_ylabel(f"95% Exclusion Limit on $\\nu_\\tau$ mass (MeV)", fontsize=15)


# Add legend
ax.legend(loc="upper right", fontsize=14, framealpha=1.0)

# Print the standard deviation on top of the plot
ax.text(
    0.05, 0.1,
    f"95% EL = {extrStdev * 1000:.1f} keV",
    #f"$\\sigma_m = {extrStdev * 1000:.1f}$ keV",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
    transform=ax.transAxes
)

if fit_type == "quadratic":
    ax.text(
        0.05, 0.05,
        f"{extrStdev / tauMass * 1e+06:.2f} ppm",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        transform=ax.transAxes
    )

# Save the figure
plt.tight_layout()
plt.savefig(f"fit_likelihood_{name}_sigma.png")
plt.savefig(f"fit_likelihood_{name}_sigma.pdf")
plt.close(fig)
