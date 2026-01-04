#!/usr/bin/env python3

import uproot
import numpy as np
import sys
import matplotlib.pyplot as plt
import awkward
import getpass

username = getpass.getuser()
inputFolder = "./outputs/treemaker/reco/"

# Configuration
if len(sys.argv) < 4:
	raise ValueError("Usage: quick-hist.py <processName> <treeFilename> <plotBranch>")

processName = sys.argv[1]
treeFilename = sys.argv[2]
plotBranch = sys.argv[3]

file = uproot.open(inputFolder + treeFilename)
if not file:
	raise FileNotFoundError("The specified ROOT file could not be opened.")

# Load the events tree
print("Loading events tree...")
events_key = None
for i in range(100):  # Check up to "events;99"
	key = f"events;{i}"
	if key in file:
		events_key = key
		break
if events_key is None:
	raise KeyError("No 'events' branch found in the file.")

events = file[events_key]

print("Picking branches...")
branches = events.arrays([
	plotBranch
], how=dict)

statistic = branches[plotBranch]

# Unwrap statistic
data = statistic
print("N = {}".format(len(data)))

print("Plotting histogram...")
plt.figure(figsize=(10, 6))
plt.hist(data, density=True, bins=150)
plt.title("Distribution of " + plotBranch + " for " + processName + f" (N = {len(data)})")
plt.yscale('log')

plt.xlabel(plotBranch)
plt.ylabel("Events")

print("Saving histogram...")
plt.savefig("./hist_" + processName + "_" + plotBranch  + ".png")
print("Histogram saved as hist_" + processName + "_" + plotBranch  + ".png")
