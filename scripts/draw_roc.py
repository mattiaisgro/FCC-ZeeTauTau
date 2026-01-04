
#
# Draw a single plot with all ROC curves
#

import matplotlib.pyplot as plt
import csv
import numpy as np

# The quark flavors to plot
flavors = ["light", "charm", "strange", "bottom"]
flavorShortened = ["ud", "c", "s", "b"]

x_list = []
y_list = []

print("Loading points from CSV files...")

for flavor in flavors:

	x = []
	y = []
	filename = "roc_curve_" + flavor + ".csv"

	with open(filename, "r") as file:
		reader = csv.reader(file)
		for row in reader:
			if len(row) == 2:

				try:
					float(row[0])
					float(row[1])
				except ValueError:
					continue

				x_val, y_val = map(float, row)
				x.append(x_val)
				y.append(y_val)

	x_list.append(x)
	y_list.append(y)

print("Plotting ROC curve...")

plt.figure(dpi=300)
for i in range(0, len(flavors)):
	plt.plot(x_list[i], y_list[i], label=f"${flavorShortened[i]}$ vs $\\tau$")

# Draw the chosen working point of 98% efficiency
plt.axvline(x=0.98, color="gray", linestyle="--", label="Working Point")

plt.xlabel("$\\tau$ Jet Efficiency", fontsize=15)
plt.xlim(0.9, 1.0)
plt.ylabel("$q$ Jet Misid.", fontsize=15)
plt.yscale("log")
plt.ylim(1E-04, 1E-02)
#plt.title("ROC Curve for $q$ Jets vs $\\tau$ Jets (N > 10^6)")
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("roc_curve.png")
plt.savefig("roc_curve.pdf")
plt.close()

print("Plotting efficiency and misid. curves...")

cuts = np.linspace(0, 1, len(x_list[0]))

plt.figure(dpi=300)
plt.plot(cuts, x_list[0], label="$\\tau$")

for i in range(0, len(flavors)):
	plt.plot(cuts, y_list[i], label=f"${flavorShortened[i]}$")

plt.xlabel("Probability Threshold", fontsize=15)
plt.ylabel("Passing Score Ratio", fontsize=15)
plt.yscale("log")
#plt.title("Efficiency and Misid. for $\\tau$ Jets (N > 10^6)")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("efficiencies.png")
plt.savefig("efficiencies.pdf")
plt.close()

print("Success!")
