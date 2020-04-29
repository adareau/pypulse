# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-29 14:58:05
Modified : 2020-04-29 17:30:08

Comments :
'''

# %% General Imports
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from numpy import pi

# %% Local Imports
from plotting import plot_pulse_core, _show_plot
from shapes import PulseShape

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

    def profile(self, t, args=''):
        '''
        Returns the full sequence profile, with all pulses.
        '''
        signal = 0
        for pulse in self.pulse_list:
            signal = signal + pulse.profile(t, args)
        return signal

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
    def plot_amp(self, t=None, ax=None, show=True, time_norm=1, amp_norm=1,
                 **kwargs):
        # get seq parameters
        T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
        Tmax = np.max(T_end)
        t0 = 0
        # call core plotting function
        ax = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                             type='amplitude', ax=ax, show=show,
                             pulse_norm=amp_norm, time_norm=time_norm,
                             **kwargs)
        return ax

    def plot_phase(self, t=None, ax=None, show=True, time_norm=1,
                   phase_norm=pi, **kwargs):
        # get seq parameters
        T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
        Tmax = np.max(T_end)
        t0 = 0
        # call core plotting function
        ax = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                             type='phase', ax=ax, show=show,
                             pulse_norm=phase_norm, time_norm=time_norm,
                             **kwargs)
        return ax

    def plot_sequence(self, t=None, ax=None, show=True, time_norm=1,
                      amp_norm=1, phase_norm=pi, **kwargs):
        # get pulse parameters
        T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
        Tmax = np.max(T_end)
        t0 = 0
        # call core plotting function
        # amplitude (show = False)
        ax = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                             type='amplitude', ax=ax, show=False,
                             pulse_norm=amp_norm, time_norm=time_norm,
                             color='C0', **kwargs)
        ax.set_ylabel('amplitude')
        ax.grid()
        # phase
        ax_phase = ax.twinx()
        ax_phase = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                                   type='phase', ax=ax_phase, show=False,
                                   pulse_norm=phase_norm, time_norm=time_norm,
                                   color='C1', **kwargs)
        ax_phase.set_ylim(-1.1, 1.1)
        ax_phase.set_ylabel('phase (units of pi)')
        _show_plot(show)
        return ax, ax_phase


# %% Tests

if __name__ == '__main__':

    rect_pulse = PulseShape(pulse_type='rect', time_offset=0)
    rect_pulse.laser_phase = pi/4

    seq = PulseSequence()
    seq.add_pulse(rect_pulse)
    seq.add_pulse(pulse_type='rect', time_offset=1.5*pi, pulse_duration=pi)
    seq.add_pulse(pulse_type='rect', time_offset=2.5*pi,
                  pulse_duration=pi, window='hanning', laser_phase=pi/6)
    seq.plot_sequence()
    U = seq.propagator()
    print(U)

    # -- plot ?
    # rect_pulse.plot_pulse(time_norm=pi)
