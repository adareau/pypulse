# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-29 15:11:25
Modified : 2020-04-29 15:58:07

Comments : Plotting functions for pypulse
'''

# %% General Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# %% Pulse shape plotting functions

# = Subroutines

def _init_plot(ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    return ax


def _show_plot(show):
    if show:
        plt.tight_layout()
        plt.show()


# = Core functions

def plot_pulse_core(profile, t=None, Tmax=0, t0=0, type='phase', ax=None,
                    show=True, time_norm=1, pulse_norm=1, **kwargs):
    # process inputs
    msg = "'type' should be 'phase' or 'amplitude'"
    assert type in ['phase', 'amplitude'], msg

    # initialize figure if needed
    ax = _init_plot(ax)
    # time
    if t is None:
        t = np.linspace(-Tmax * 0.05, Tmax * 1.05, 1000) + t0
    # plot
    if type == 'amplitude':
        pulse_intensity = np.abs(profile(t, '')) ** 2
        ax.plot(t / time_norm, pulse_intensity / pulse_norm, **kwargs)
    elif type == 'phase':
        pulse_phase = np.angle(profile(t, ''))
        ax.plot(t / time_norm, pulse_phase / pulse_norm, **kwargs)

    # show if neede
    _show_plot(show)

    return ax
