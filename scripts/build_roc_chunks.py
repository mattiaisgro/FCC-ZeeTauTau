
#
# Compute the ROC curve, starting from two datasets
# containing scores for true events and false events,
# using chunks instead of a single file.
#

import uproot
import awkward
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import concurrent.futures
import getpass

if len(sys.argv) < 2:
	raise ValueError("Usage: build_roc_chunks.py <flavor>")

# Flavor to build the ROC curve of
flavor = sys.argv[1]

# How many curve points to compute
M = 1000

# Name of the branch containing true events
trueBranch = "jets_tau_score"

# Name of the branch containing false events
falseBranch = "jets_tau_score"

# The maximum number of files to read
maxFiles = 1


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


# List the ROOT files inside a folder
def get_root_files(folder):
	files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".root")]
	return files[:maxFiles]

# Load the scores from each file
def load_scores(filename, branch_name):
	try:
		file = uproot.open(filename)
		events = file["events;1"]
		print("Branches: " + str(events.keys()))
		branches = events.arrays([branch_name], how=dict)
		scores = awkward.flatten(branches[branch_name])
		return scores
	except Exception as e:
		print(f"Error reading {filename}: {e}")
		return awkward.Array([])

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


# Folders containing the ROOT files
username = getpass.getuser()
tauFolder = f"/eos/user/{username[0]}/{username}/Ztautau/efficiency/hadronic/p8_ee_Ztautau_ecm91/"
quarkFolder = f"/eos/user/{username[0]}/{username}/Ztautau/efficiency/hadronic/p8_ee_Z" + products +"_ecm91/"

# Get all ROOT files in the folders
tauFiles = get_root_files(tauFolder)
quarkFiles = get_root_files(quarkFolder)

print(f"Found {len(tauFiles)} tau files and {len(quarkFiles)} quark files.")

if len(tauFiles) == 0 or len(quarkFiles) == 0:
	print("No ROOT files found!")
	exit(1)

# Load tau flavor scores in parallel
print("Loading tau events from all files...")
with concurrent.futures.ThreadPoolExecutor() as executor:
	tauScoresList = list(executor.map(lambda f: load_scores(f, trueBranch), tauFiles))

tauScores = awkward.flatten(tauScoresList)
print("N_true = " + str(len(tauScores)))

# Load quark flavor scores in parallel
print("Loading quark events from all files...")
with concurrent.futures.ThreadPoolExecutor() as executor:
	quarkScoresList = list(executor.map(lambda f: load_scores(f, falseBranch), quarkFiles))

quarkScores = awkward.flatten(quarkScoresList)
print("N_false = " + str(len(quarkScores)))


# Compute M points between 0 and 1
cuts = np.linspace(0, 1, M)
x = []
y = []

tauEfficiency = []
quarkMisid = []

print("Computing ROC curve...")

tauScores = np.sort(tauScores)
quarkScores = np.sort(quarkScores)

for cut in cuts:
	new_x, new_y = ROC_point(tauScores, quarkScores, cut)

	x.append(new_x)
	y.append(new_y)

	tauEfficiency.append(new_x)
	quarkMisid.append(new_y)

print("Saving the data points to file...")

with open(f"roc_curve_{flavor}.csv", "w") as ROCFile:
	ROCFile.write("\"Tau Efficiency\", \"" + flavor.capitalize() +" Quark Misid.\" \n")
	for xi, yi in zip(x, y):
		ROCFile.write(f"{xi},{yi}\n")


print("Plotting individual curves...")

plt.figure(figsize=(10, 6))
plt.plot(cuts, tauEfficiency, label="Tau Tagging Efficiency")
plt.xlabel("Probability Threshold")
plt.xlim(0.92, 1.0)
plt.ylabel("Tau Jet Efficiency")
plt.title("Tau Tagging Efficiency (N_true = " + str(len(tauScores)) + ")")
plt.grid(True)
plt.savefig("tau_efficiency.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(cuts, quarkMisid, label=flavor.capitalize() + "Quark Jet Misidentification")
plt.xlabel("Probability Threshold")
plt.ylabel("Quark Jet Misidentification")
plt.yscale("log")
plt.title("Quark Jet Misidentification (N_false = " + str(len(quarkScores)) + ")")
plt.grid(True)
plt.savefig("quark_misid_" + flavor +".png")
plt.close()

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
