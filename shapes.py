# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-27 14:27:18
Modified : 2020-04-29 15:03:37

Comments :
'''

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# %% Functions


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
        self.auto_amplitude = False
        self.sinc_minima = 8  # number of minima for sinc pulse

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
        pulse_types = ['RECT', 'SINC']
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

        # -- compute window
        if self.window is None:
            window = self.square(tc / T)
        elif self.window.upper() == 'HANNING':
            window = self.hanning(tc / T)
        elif self.window.upper() == 'BLACKMAN':
            window = self.blackman(tc / T)

        return x * window

    # --- Methods : analyse and plot

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

    def plot_amp(self, t=None, ax=None, show=True, t_norm=1, **kwargs):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # time
        if t is None:
            T = self.pulse_duration
            t0 = self.time_offset
            t = np.linspace(-T * 0.05, T * 1.05, 1000) + t0
        # plot
        pulse_intensity = np.abs(self.profile(t, '')) ** 2
        ax.plot(t / t_norm, pulse_intensity, **kwargs)
        # show if neede
        self._show_plot(show)
        return ax

    def plot_phase(self, t=None, ax=None, show=True, t_norm=1,
                   phi_norm=pi, **kwargs):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # time
        if t is None:
            T = self.pulse_duration
            t0 = self.time_offset
            t = np.linspace(-T * 0.05, T * 1.05, 1000) + t0
        # plot
        pulse_phase = np.angle(self.profile(t, ''))
        ax.plot(t / t_norm, pulse_phase / phi_norm, **kwargs)
        # show if neede
        self._show_plot(show)
        return ax

    def plot_pulse(self, t=None, ax=None, show=True, t_norm=1):
        # initialize figure if needed
        ax = self._init_plot(ax)
        # plot amplitude
        self.plot_amp(t, ax, False, t_norm, color='C0')
        ax.set_ylabel('amplitude')
        ax.grid()
        # plot phase
        ax_phase = ax.twinx()
        self.plot_phase(t, ax_phase, False, t_norm, pi, color='C1')
        ax_phase.set_ylim(-1.1, 1.1)
        ax_phase.set_ylabel('phase (units of pi)')
        # show if neede
        self._show_plot(show)
        return ax


# %% Tests

if __name__ == '__main__':

    rect_pulse = PulseShape(pulse_type='rect', time_offset=0)
    rect_pulse.plot_pulse(t_norm=pi)
