
import ROOT
from matplotlib import pyplot as plt
import numpy as np


# New Reco - Tau
#chi2 = [ 0, ,  ]
#name = "Mtau_reco_new"

# Old Reco - Tau
#chi2 = [ 0, ,  ]
#name = "Mtau_reco_old"


# New Reco - Neutrino
#chi2 = [ 0, ,  ]
#name = "Mnutau_reco_new"

# Old Reco - Neutrino
#chi2 = [ 0, ,  ]
#name = "Mnutau_reco_old"


# New Reco - Tau 5-Prongs
chi2 = [ 0, 1273380.2600412206, 1231936.5338207101 ]
name = "Mtau_reco_new_5prongs"

# Old Reco - Tau 5-Prongs
chi2 = [ 0, 685285.5546284923, 950953.4225386115 ]
name = "Mtau_reco_old_5prongs"


# New Reco - Neutrino 5-Prongs
chi2 = [ 0, 719309.943096773, 21625109.41363103 ]
name = "Mnutau_reco_new_5prongs"

# Old Reco - Neutrino 5-Prongs
chi2 = [ 0, 834368.9606239314, 762766.9139313011 ]
name = "Mnutau_reco_old_5prongs"


displayName = name.replace("_", " ")

# Cut on the chi2 curve (1 = 1*sigma, 4 = 2*sigma, 2.71 = 2-sigma 1-sided ...)
#chi2Cut = 1
chi2Cut = 2.71

# Number of events in MC sample
Nmc = 1e+8

# Cross section in the sample
crossSection = 1476.58 # pb

# Luminosity such that N(MC) = N(Data)
Leq = (Nmc / crossSection) * 1e-06


# X values
#massVariations = [ 0, -10, +10 ]  # Tau Mass variations
massVariations = [ 0, +10, +100 ] # Neutrino mass variations


# Total integrated luminosity
luminosityTotal = 205 # /ab

# Mass of the tau lepton
tauMass = 1776.93 # Mev

# Luminosity reduction factor
luminosityFactors = [ 2**(i-30) for i in range(0, 32) ]
print("Luminosity Factors = " + str(luminosityFactors))


lumValues = []
sigmaValues = []

for i in range(0, len(luminosityFactors)):

    luminosityFactor = luminosityFactors[i]

    print(f"Processing luminosity {luminosityTotal * luminosityFactor}")

    # Example data (replace with your own)
    x_vals = np.array(massVariations, dtype=float)
    y_vals = np.array(chi2, dtype=float) * luminosityFactor

    # Create a TGraph
    n_points = len(x_vals)
    graph = ROOT.TGraph(n_points, x_vals, y_vals)

    # Fit with a quadratic function (pol2)
    fit_result = graph.Fit("pol2", "S")
    fit_params = fit_result.GetParams()
    fit_errors = [fit_result.ParError(i) for i in range(3)]

    # Optional: Draw the graph and fit
    canvas = ROOT.TCanvas("c"+str(i), "Quadratic Fit", 1600, 1200)
    canvas.SetLeftMargin(0.15)
    graph.SetTitle(f"Mass Sensibility Fit ({displayName}, Luminosity = {luminosityFactor * luminosityTotal} /ab);Mass Variation (MeV);Chi-Squared")
    graph.SetMarkerStyle(20)
    graph.Draw("AP")

    # Draw fit parameters on the canvas
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.03)

    latex.DrawLatex(0.27, 0.86, f"a = {fit_params[0]:.3f} #pm {fit_errors[0]:.3f}")
    latex.DrawLatex(0.27, 0.80, f"b = {fit_params[1]:.3f} #pm {fit_errors[1]:.3f}")
    latex.DrawLatex(0.27, 0.74, f"c = {fit_params[2]:.3f} #pm {fit_errors[2]:.3f}")

    # Compute intersections between the parabola and y=1
    a, b, c = fit_params[0], fit_params[1], fit_params[2]
    coeffs = [c, b, a - chi2Cut]
    roots = np.roots(coeffs)

    x0 = roots[0].real
    x1 = roots[1].real
    sigma = max(abs(x0), abs(x1))

    latex.DrawLatex(0.27, 0.68, f"#sigma = {sigma:.6f} MeV")

    # Save the canvas
    if i == (len(luminosityFactors) - 2):
        canvas.SaveAs(f"fit_chi2_{name}_mass.png")

    print(f"\nluminosity = {luminosityFactor * luminosityTotal:.2f}, sigma = {sigma:.2f}\n")
    lumValues.append(luminosityFactor * luminosityTotal)
    sigmaValues.append(sigma)


# Extrapolate values following inverse law
extrValues = []
scalarFactor = sigmaValues[0] * (luminosityTotal * luminosityFactors[0])**0.5

for luminosityFactor in luminosityFactors:
    extrValues.append(scalarFactor / (luminosityFactor * luminosityTotal)**0.5)

# Extrapolated standard deviation for the integrated luminosity
extrStdev = scalarFactor / luminosityTotal**0.5

# Convert lumValues and sigmaValues to ROOT arrays
lumArray = np.array(lumValues, dtype=float)
sigmaArray = np.array(sigmaValues, dtype=float) #/ tauMass * 1e+06
extrArray = np.array(extrValues, dtype=float) #/ tauMass * 1e+06


# Create Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_ybound(1, 1e+05)

# Plot fit graph
ax.plot(lumArray, sigmaArray, marker='o', color='blue', linewidth=2, label='Asimov fit')

# Plot extrapolation graph
ax.plot(lumArray, extrArray, linestyle='--', color='orange', linewidth=2, label='Statistical scaling')

# Draw vertical red lines at x = Leq/100 and x = 205
ax.axvline(Leq / 100, color='red', linewidth=2, label='$N_{MC} = 100 N_{data}$')
ax.axvline(Leq / 10, color='orange', linewidth=2, label='$N_{MC} = 10 N_{data}$')
ax.axvline(205, color='green', linewidth=2, label='FCC-ee Z pole run')

# Set axis labels and title
ax.set_title(f"Mass Stat. Uncertainty vs Luminosity ({displayName})")
ax.set_xlabel("Luminosity ($ab^{-1}$)")

ax.set_ylabel(f"95% Exclusion Limit (MeV)")
#ax.set_ylabel(f"Statistical $\\sigma_m$ (ppm)")


# Add legend
ax.legend(loc="upper right", fontsize=12, framealpha=1.0)

# Print the standard deviation on top of the plot
ax.text(
    0.05, 0.1,
    f"95% EL = {extrStdev * 1000:.1f} keV",
    #f"$\\sigma_m = {extrStdev * 1000:.1f}$ keV",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
    transform=ax.transAxes
)

#ax.text(
#    0.05, 0.05,
#    f"{extrStdev / tauMass * 1e+06:.2f} ppm",
#    fontsize=12,
#    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
#    transform=ax.transAxes
#)

# Save the figure
plt.tight_layout()
plt.savefig(f"fit_chi2_{name}_sigma.png")
plt.savefig(f"fit_chi2_{name}_sigma.pdf")
plt.close(fig)
