#!/usr/bin/env python3


import sys
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from matplotlib.animation import FuncAnimation, PillowWriter


if len(sys.argv) != 2:
    print("usage: ./confusion_vis.py <CONFUSION_JSON>")
    sys.exit(1)

file = sys.argv[1]


def get_confusion_data(f):
    with open(f, "r") as cfs:
        return json.load(cfs)["data"]


data = get_confusion_data(file)

fig, ax = plt.subplots()


def animate(i):
    means, variances = data[i]
    std_devs = np.sqrt(np.asarray(variances))
    ax.clear()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("Mean and Standard Deviation, 128 Epochs")
    l1 = ax.plot(range(-34, 47), range(-34, 47))
    l2 = ax.errorbar(range(-34, 47), np.round(means), yerr=std_devs, fmt='o', elinewidth=1, ms=2)
    return l1, l2


ani = FuncAnimation(fig, animate, interval=40, repeat=True, frames=len(data))
ani.save("learning.gif", dpi=300, writer=PillowWriter(fps=20))
