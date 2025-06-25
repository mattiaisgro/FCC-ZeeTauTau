
#
# Draw a single plot with all ROC curves
#

import matplotlib.pyplot as plt
import csv
import numpy as np

# The quark flavors to plot
flavors = ["light", "charm", "strange", "bottom"]

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

plt.figure()
for i in range(0, len(flavors)):
	plt.plot(x_list[i], y_list[i], label=f"{flavors[i].capitalize()}")

plt.xlabel("Tau Jet Efficiency")
plt.xlim(0.94, 1.0)
plt.ylabel("Quark Jet Misid.")
plt.yscale("log")
plt.yscale()
plt.title("ROC Curve for Quark Jets vs Tau Jets (N > 10^3)")
plt.legend()
plt.grid(True)

plt.savefig("roc_curve.png")
plt.close()

print("Plotting efficiency and misid. curves...")

cuts = np.linspace(0, 1, len(x_list[0]))

plt.figure()
plt.plot(cuts, x_list[0], label="Tau Jet Efficiency")

for i in range(0, len(flavors)):
	plt.plot(cuts, y_list[i], label=f"{flavors[i].capitalize()} Misid.")

plt.xlabel("Probability Threshold")
plt.ylabel("Passing Score Ratio")
plt.yscale("log")
plt.title("Efficiency and Misid. for Tau Jets (N > 10^3)")
plt.legend()
plt.grid(True)
plt.savefig("efficiencies.png")
plt.close()

print("Success!")
