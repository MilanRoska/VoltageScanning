# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:53:11 2026

@author: m.roska

Matplotlib style parameters and my custom colors.
Because who doesnt want custom colors?!
Arent they pretty? :)

"""

# %% packages

import matplotlib.pyplot as plt


# %% functions

def standard_colors():
    orange = (244/255, 153/255, 62/255, 100/100)
    orange_light = (252/255, 230/255, 207/255, 100/100)
    orange_dark = (145/255, 77/255, 8/255, 100/100)
    orange_very_dark = (73/255, 39/255, 4/255, 100/100)
    purple = (215/255, 61/255, 245/255, 100/100)
    purple_light = (252/255, 240/255, 254/255, 100/100)
    purple_dark = (122/255, 8/255, 145/255, 100/100)
    purple_very_dark = (61/255, 4/255, 73/255, 100/100)
    return orange, orange_light, orange_dark, orange_very_dark, purple, purple_light, purple_dark, purple_very_dark


def standard_plot_parameters(ax, dark_mode=False):
    plt.tight_layout()
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.legend(loc='upper right', handlelength=1, scatterpoints=1)
    ax.tick_params(axis='both', direction='in', length=5, width=1)
    ax.tick_params(which='major', size=5)  # Major ticks
    ax.tick_params(which='minor', size=5)   # Minor ticks
    # Show ticks on all four sides of the plot
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    if dark_mode:
        plt.style.use('dark_background')
        plt.style.use('classic')
    return
