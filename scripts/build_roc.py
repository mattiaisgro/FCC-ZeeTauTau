
#
# Compute the ROC curve, starting from two datasets
# containing scores for true events and false events
#

import uproot
import awkward
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	raise ValueError("Usage: build_ROC_curve.py <flavor>")

# Flavor to build the ROC curve of
flavor = sys.argv[1]

# How many curve points to compute
M = 10000

# Name of the branch containing true events
trueBranch = "jets_tau_score"

# Name of the branch containing false events
falseBranch = "jets_" + flavor + "_tau_score"


products = ""
flavorShortened = ""

if flavor == "light":
    products = "ud"
    flavorShortened = "l"
elif flavor == "charm":
    products = "cc"
    flavorShortened = "c"
elif flavor == "strange":
    products = "ss"
    flavorShortened = "s"
elif flavor == "bottom":
    products = "bb"
    flavorShortened = "b"

tauFilename = "../outputs/treemaker/Ztautau/hadronic/p8_ee_Ztautau_ecm91.root"
quarkFilename = "../outputs/treemaker/Zqq/hadronic/p8_ee_Z" + products + "_ecm91.root"


# Indices for scores
i_true = 0
i_false = 0

# Compute the rate of true positives
def true_cut(true_scores, cut):

	global i_true

	while i_true < len(true_scores) and true_scores[i_true] < cut:
		i_true += 1

	return len(true_scores) - i_true


# Compute the rate of false positives
def false_cut(false_scores, cut):

	global i_false

	while i_false < len(false_scores) and false_scores[i_false] < cut:
		i_false += 1

	return len(false_scores) - i_false

# Compute a single point of the ROC curve
# from the scores and the cut
def ROC_point(true_scores, false_scores, cut):

	x = true_cut(true_scores, cut) / len(true_scores)
	y = false_cut(false_scores, cut) / len(false_scores)

	return x , y


# Load tau flavor scores
tauFile = uproot.open(tauFilename)
if not tauFile:
	raise FileNotFoundError("The tau ROOT file could not be opened.")

print("Loading tau events tree...")
tauEvents = tauFile["events;1"]
tauBranches = tauEvents.arrays([
	"jets_tau_score"
], how=dict)
tauScores = awkward.flatten(tauBranches["jets_tau_score"])
print("N_true = " + str(len(tauScores)))


# Load quark flavor scores
quarkFile = uproot.open(quarkFilename)
if not quarkFile:
	raise FileNotFoundError("The tau ROOT file could not be opened.")

print("Loading tau events tree...")
quarkEvents = quarkFile["events;1"]
quarkBranches = quarkEvents.arrays([
	"jets_" + flavor + "_tau_score"
], how=dict)
quarkScores = awkward.flatten(quarkBranches["jets_" + flavor + "_tau_score"])
print("N_false = " + str(len(quarkScores)))

# Compute M points between 0 and 1
cuts = np.linspace(0, 1, M)
x = []
y = []

print("Computing ROC curve...")

tauScores = np.sort(tauScores)
quarkScores = np.sort(quarkScores)

for cut in cuts:
	new_x, new_y = ROC_point(tauScores, quarkScores, cut)
	x.append(new_x)
	y.append(new_y)

print("Saving the data points to file...")

with open(f"roc_curve_{flavor}.csv", "w") as ROCFile:
	ROCFile.write("\"Tau Efficiency\", \"" + flavor.capitalize() +" Quark Misid.\" \n")
	for xi, yi in zip(x, y):
		ROCFile.write(f"{xi},{yi}\n")

print("Plotting ROC curve...")

plt.figure()
plt.plot(x, y, label=f"ROC Curve for " + flavor.capitalize() + " Quark Jets")
plt.xlabel("Tau Jet Efficiency (N_true = " + str(len(tauScores)) + ")")
plt.ylabel(flavor.capitalize() + " Quark Jet Misid. (N_false = " + str(len(quarkScores)) + ")")
plt.yscale("log")
plt.title(f"ROC Curve for {flavor.capitalize()} Quark Jets")
plt.legend()
plt.grid(True)
plt.savefig(f"roc_curve_{flavor}.png")
plt.close()

print("Success!")
