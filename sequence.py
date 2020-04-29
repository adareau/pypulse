# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-29 14:58:05
Modified : 2020-04-29 15:07:08

Comments :
'''

# %% General Imports
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from numpy import pi

# %% Local Imports
from .shapes import PulseShape


# %% Class

class PulseSequence():
    '''
    This class will allow to define and study a sequence
    of Bragg pulses.
    '''

    def __init__(self, **kwargs):
        '''
        Object initialization, sets parameters
        '''

        # -- initialize default settings
        # physical parameters
        self.global_detuning = 0  # diffraction detuning
        self.pulse_list = []  # will contain a list of PulseShape objects

        # -- initialize object
        # update attributes based on kwargs
        self.__dict__.update(kwargs)

        # -- initialize default settings

        # hidden attributes
        self._print = True

    # --- Hidden subroutines

    def _p(self, s):
        '''
        used to easily enable/disable all written output
        '''
        if self._print:
            print(s)

    # --- Methods : calculations and definitions

    def reset_pulse_list(self):
        self.pulse_list = []

    def add_pulse(self, pulse=None, **kwargs):
        '''
        Adds a pulse profile to the analyzer. Can be called in two ways :
        + Version #1 :
            > add_pulse(pulse_shape)
            where `pulse_shape` is a PulseShape object. Then `pulse_shape` is
            appended to the pulse_list

        + Version #2 :
            > add_pulse(**kwargs)
            creates a new PulseShape object. **kwargs are passed to the
            PulseShape generator __init__ function.
        '''
        if isinstance(pulse, PulseShape):
            self.pulse_list.append(pulse)
        else:
            pulse = PulseShape(**kwargs)
            self.pulse_list.append(pulse)

    def propagator(self, T=None):
        '''
        Computes the propagator (evolution operator) for the pulse
        '''
        # -- get parameters
        delta = self.global_detuning
        if T is None:
            T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
            T = np.max(T_end)
            msg = 'No time provided, use total duration (T=%.2f)' % T
            self._p(msg)

        # -- define Hamiltonian
        # free Hamiltonian
        H0 = 0.5 * delta * qt.sigmaz()
        H = [H0]
        # pulses
        Hr_imag = 0.5 * qt.sigmay()
        Hr_real = 0.5 * qt.sigmax()
        for pulse in self.pulse_list:
            H.append([Hr_real, pulse._pulse_real])
            H.append([Hr_imag, pulse._pulse_imag])

        # -- compute propagator and return
        U = qt.propagator(H, T)

        return U

    # --- Methods : analyse and plot

    # - PLOTTING
    @staticmethod
    def _init_plot(ax):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        return ax

    @staticmethod
    def _show_plot(show):
        if show:
            plt.tight_layout()
            plt.show()

    def plot_pulse_amp(self, t=None, ax=None, show=True, t_norm=1, **kwargs):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # time
        if t is None:
            T = self.pulse_duration
            t = np.linspace(-T * 0.05, T * 1.05, 1000)
        # plot
        pulse_intensity = np.abs(self.pulse_profile(t, '')) ** 2
        ax.plot(t / t_norm, pulse_intensity, **kwargs)
        # show if neede
        self._show_plot(show)
        return ax

    def plot_pulse_phase(self, t=None, ax=None, show=True, t_norm=1,
                         phi_norm=pi, **kwargs):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # time
        if t is None:
            T = self.pulse_duration
            t = np.linspace(-T * 0.05, T * 1.05, 1000)
        # plot
        pulse_phase = np.angle(self.pulse_profile(t, ''))
        ax.plot(t / t_norm, pulse_phase / phi_norm, **kwargs)
        # show if neede
        self._show_plot(show)
        return ax

    def plot_pulse(self, t=None, ax=None, show=True, t_norm=1):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # plot amplitude
        self.plot_pulse_amp(t, ax, False, t_norm, color='C0')
        ax.set_ylabel('amplitude')
        ax.grid()
        # plot phase
        ax_phase = ax.twinx()
        self.plot_pulse_phase(t, ax_phase, False, t_norm, pi, color='C1')
        ax_phase.set_ylim(-1.1, 1.1)
        ax_phase.set_ylabel('phase (units of pi)')
        # show if neede
        self._show_plot(show)
        return ax


# %% Tests

if __name__ == '__main__':

    rect_pulse = PulseShape(pulse_type='rect', time_offset=0)
    rect_pulse.laser_phase = pi/4

    seq = PulseSequence()
    seq.add_pulse(rect_pulse)
    seq.add_pulse(pulse_type='rect', time_offset=pi, pulse_duration=pi)
    U = seq.propagator()
    print(U)

    # -- plot ?
    rect_pulse.plot_pulse(t_norm=pi)
