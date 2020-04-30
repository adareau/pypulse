# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-27 14:27:18
Modified : 2020-04-30 16:22:59

Comments :
'''

# %% General Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# %% Local Imports
from plotting import plot_pulse_core, _show_plot

# %% Class


class PulseShape():
    '''
    Defines pulse shapes (temporal)
    '''
    def __init__(self, **kwargs):
        '''
        Object initialization, sets parameters
        '''
        # -- initialize default settings
        # physical parameters
        self.detuning = 0  # TODO: include in pulseshape
        self.laser_phase = 0
        self.rabi_pulsation = 1
        self.pulse_duration = pi
        self.time_offset = 0

        # pulse shape
        self.pulse_type = 'RECT'  # RECT|SINC
        self.window = None
        self.auto_amplitude = False  # FIXME : not implemented
        self.sinc_minima = 8  # number of minima for sinc pulse
        self.phi_sinc = 0

        # -- initialize object
        # update attributes based on kwargs
        self.__dict__.update(kwargs)
        # check parameters
        self._check_params()

    # -- Hidden subroutines

    def _check_params(self):
        '''
        checks whether current parameters are OK
        '''
        # is pulse type implemented ?
        pulse_types = ['RECT', 'SINC', 'SSINC']
        msg = 'Pulse type not implemented. List of implemented pulse types : '
        msg += ', '.join(pulse_types)
        assert self.pulse_type.upper() in pulse_types, msg

        # is window type implemented ?
        window_types = ['HANNING', 'BLACKMAN']
        msg = 'Pulse type not implemented. List of implemented pulse types : '
        msg += ', '.join(window_types)
        if self.window is not None:
            assert self.window.upper() in window_types, msg

    def _pulse_real(self, t, args):
        '''
        shorthand to get real part of the pulse profile
        '''
        return np.real(self.profile(t, args))

    def _pulse_imag(self, t, args):
        '''
        shorthand to get real part of the pulse profile
        '''
        return np.imag(self.profile(t, args))

    # -- Windows

    @staticmethod
    def square(x):
        in_pulse = np.logical_and(x > 0, x < 1)
        return np.where(in_pulse, 1, 0)

    @staticmethod
    def hanning(x):
        in_pulse = np.logical_and(x > 0, x < 1)
        out = 0.5 - 0.5 * np.cos(2*pi*x)
        return np.where(in_pulse, out, 0)

    @staticmethod
    def blackman(x):
        in_pulse = np.logical_and(x > 0, x < 1)
        a0 = 7938/18608
        a1 = 9240/18608
        a2 = 1430/18608
        out = a0 - a1 * np.cos(2*pi*x) + a2 * np.cos(4*pi*x)
        return np.where(in_pulse, out, 0)

    # --- Profiles
    @staticmethod
    def ssinc(x, phi=0):
        '''
        shifted sinc !
        '''
        x = np.asanyarray(x, dtype=np.float)
        x = np.pi * x
        x = np.where(x == 0, 1e-20, x)
        y = (np.sin(x - 0.5 * phi) + np.sin(0.5 * phi)) / x
        y = np.where(x == 1e-20, np.cos(0.5 * phi), y)
        return y

    # --- Methods : calculations

    def profile(self, t, args):
        '''
        Returns temporal pulse profile (complex !)
        '''
        # -- get parameters
        OmegaR = self.rabi_pulsation
        phi = self.laser_phase
        T = self.pulse_duration
        t0 = self.time_offset

        # -- compute profile
        tc = t-t0
        if self.pulse_type.upper() == 'RECT':
            x = OmegaR * np.exp(1j * phi)
        elif self.pulse_type.upper() == 'SINC':
            Nm = self.sinc_minima
            Os = 2 * np.pi * Nm / T  # sinc frequency
            x = OmegaR * np.sinc(Os * (tc - T / 2) / np.pi) * np.exp(1j * phi)
        elif self.pulse_type.upper() == 'SSINC':
            Nm = self.sinc_minima
            phi_sinc = self.phi_sinc
            Os = 2 * np.pi * Nm / T  # sinc frequency
            x = OmegaR * self.ssinc(Os * (tc - T / 2) / np.pi, phi_sinc)
            x = x * np.exp(1j * phi)

        # -- compute window
        if self.window is None:
            window = self.square(tc / T)
        elif self.window.upper() == 'HANNING':
            window = self.hanning(tc / T)
        elif self.window.upper() == 'BLACKMAN':
            window = self.blackman(tc / T)

        return x * window

    # --- Methods : analyse and plot

    def plot_amp(self, t=None, ax=None, show=True, time_norm=1, amp_norm=1,
                 **kwargs):
        # get pulse parameters
        Tmax = self.pulse_duration
        t0 = self.time_offset
        # call core plotting function
        ax = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                             type='amplitude', ax=ax, show=show,
                             pulse_norm=amp_norm, time_norm=time_norm,
                             **kwargs)
        return ax

    def plot_phase(self, t=None, ax=None, show=True, time_norm=1,
                   phase_norm=pi, **kwargs):
        # get pulse parameters
        Tmax = self.pulse_duration
        t0 = self.time_offset
        # call core plotting function
        ax = plot_pulse_core(self.profile, t=t, Tmax=Tmax, t0=t0,
                             type='phase', ax=ax, show=show,
                             pulse_norm=phase_norm, time_norm=time_norm,
                             **kwargs)
        return ax

    def plot_pulse(self, t=None, ax=None, show=True, time_norm=1, amp_norm=1,
                   phase_norm=pi, **kwargs):
        # get pulse parameters
        Tmax = self.pulse_duration
        t0 = self.time_offset
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

    rect_pulse = PulseShape(pulse_type='ssinc', time_offset=0, phi_sinc=pi/2)
    rect_pulse.laser_phase = pi/4
    rect_pulse.plot_pulse(time_norm=pi)
