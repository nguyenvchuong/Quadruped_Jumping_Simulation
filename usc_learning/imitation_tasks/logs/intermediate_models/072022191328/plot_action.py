import matplotlib.pyplot as plt
import csv
import numpy as np

action_file = "actions_0.csv"

fl_x_command = []

with open(action_file) as f:
    reader = csv.reader(f)
    for row in reader:
        fl_x_command.extend(row[::12])

fl_x_command = list(map(float, fl_x_command))
x = np.linspace(0.02, 0.9, num=len(fl_x_command))
plt.plot(x, fl_x_command)
plt.xlabel("Time (s)")
plt.ylabel("Front left leg x command")
plt.show()
