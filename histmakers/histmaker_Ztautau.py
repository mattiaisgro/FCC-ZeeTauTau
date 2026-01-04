#!/usr/bin/env python3

import uproot 
import numpy as np
import matplotlib.pyplot as plt

# Configuration
processName = "Ztautau"
plotBranch = "jets_R5_tau_true_number"
treeFilename = "hadronic/p8_ee_Ztautau_ecm91.root"


file = uproot.open("../outputs/treemaker/" + processName + "/" + treeFilename)
if not file:
	raise FileNotFoundError("The specified ROOT file could not be opened.")

# Load the events tree
print("Loading events tree...")
events = file["events;1"]

print("Picking branches...")
branches = events.arrays([
	plotBranch
], how=dict)

statistic = branches[plotBranch]

print("Plotting histogram...")
plt.figure(figsize=(10, 6))
plt.hist(statistic, bins=np.arange(0, 10) - 0.5, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title("Number of Tau Jets")

plt.xlabel("Number of Jets")
plt.ylabel("Normalized Frequency")

print("Saving histogram...")
plt.savefig("./hist_" + processName  +".png")
print("Histogram saved as hist_" + processName + ".png")
