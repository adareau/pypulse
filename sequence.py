# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-04-29 14:58:05
Modified : 2020-05-07 18:15:05

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
        self.decoherence_rate = 0  # "T2"-decoherence rate #TODO !!
        self.pulse_list = []  # will contain a list of PulseShape objects

        # -- initialize object
        # update attributes based on kwargs
        self.__dict__.update(kwargs)

        # -- initialize default settings

        # hidden attributes
        self._verbose = False

    # --- Hidden subroutines

    def _p(self, s):
        '''
        used to easily enable/disable all written output
        '''
        if self._verbose:
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

    def propagator(self, T=None, free=False, ode_options={}):
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

        # pulses
        if not free:
            H = [H0]
            Hr_imag = 0.5 * qt.sigmay()
            Hr_real = 0.5 * qt.sigmax()
            for pulse in self.pulse_list:
                H.append([Hr_real, pulse._pulse_real])
                H.append([Hr_imag, pulse._pulse_imag])
        else:
            H = H0

        # decoherence
        if self.decoherence_rate != 0:  # FIXME: not working yet
            gamma = self.decoherence_rate
            c_ops = [np.sqrt(0.5 * gamma) * qt.sigmaz()]
        else:
            c_ops = []

        # -- compute propagator and return
        U = qt.propagator(H, T, c_op_list=c_ops, options=qt.Options(**ode_options))

        return U

    def diffraction_matrix(self, T=None, ode_options={}):
        '''
        Computes the diffraction matrix (without the dynamical phase) !
        '''
        # -- set duration if not provided
        if T is None:
            T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
            T = np.max(T_end)
            msg = 'No time provided, use total duration (T=%.2f)' % T
            self._p(msg)

        # -- compute propagation
        # with pulses
        U = self.propagator(T, ode_options=ode_options)
        # free
        U0 = self.propagator(0.5 * T, free=True, ode_options=ode_options)

        # -- compute matrix
        D = U0.dag() * U * U0.dag()

        return D

    def time_evolution(self, T=None, ode_options={}):
        '''
        Computes times evolution of the system
        '''
        # -- get parameters
        delta = self.global_detuning
        if T is None:
            T_end = [p.time_offset + p.pulse_duration for p in self.pulse_list]
            T_end = np.max(T_end)
            T = np.linspace(0, T_end, 200)

        # -- compute
        # free Hamiltonian
        H0 = 0.5 * delta * qt.sigmaz()

        # pulses
        H = [H0]
        Hr_imag = 0.5 * qt.sigmay()
        Hr_real = 0.5 * qt.sigmax()
        for pulse in self.pulse_list:
            H.append([Hr_real, pulse._pulse_real])
            H.append([Hr_imag, pulse._pulse_imag])

        # decoherence
        if self.decoherence_rate != 0:
            gamma = self.decoherence_rate
            c_ops = [np.sqrt(0.5 * gamma) * qt.sigmaz()]
        else:
            c_ops = []

        # propagator
        U = qt.propagator(H, T, c_op_list=c_ops, options=ode_options)

        # -- get phase and amplitude
        amp = {}
        phase = {}
        for i in [0, 1]:
            for j in [0, 1]:
                key = '%i%i' % (i, j)
                amp[key] = np.array([np.abs(u[i, j]) ** 2 for u in U])
                phase[key] = np.array([np.angle(u[i, j]) for u in U])

        phase['R'] = phase['01'] - phase['10']
        phase['T'] = phase['00'] - phase['11']
        amp['R'] = amp['01']
        amp['T'] = amp['00']

        result = {'amplitude': amp,
                  'phase': phase,
                  'time': T}

        return result

    def get_phase_and_amp(self, T=None, delta=None, nodyn=True,
                          ode_options={}):
        '''
        Computes phase and amplitude of the diffraction matrix (nodyn=True)
        or the propagator (nodyn=False). Either for a single detuning (delta)
        or for a list.
        '''
        # -- prepare delta
        delta_save = self.global_detuning
        if delta is None:
            delta = self.global_detuning
        if not isinstance(delta, (list, np.ndarray)):  # FIXME : moche..
            delta = np.array([delta])

        # -- compute
        M = []
        for d in delta:
            self.global_detuning = d
            if nodyn:
                M.append(self.diffraction_matrix(T, ode_options=ode_options))
            else:
                M.append(self.propagator(T, ode_options=ode_options))

        # -- get phase and amplitude
        amp = {}
        phase = {}
        for i in [0, 1]:
            for j in [0, 1]:
                key = '%i%i' % (i, j)
                amp[key] = np.array([np.abs(m[i, j]) ** 2 for m in M])
                phase[key] = np.array([np.angle(m[i, j]) for m in M])

        phase['R'] = phase['01'] - phase['10']
        phase['T'] = phase['00'] - phase['11']
        amp['R'] = amp['01']
        amp['T'] = amp['00']

        result = {'amplitude': amp,
                  'phase': phase,
                  'delta': delta}

        # -- restore
        self.global_detuning = delta_save

        return result

    # --- Methods : analyse and plot

    # - PLOTTING

    def plot_seq_amp(self, t=None, ax=None, show=True, time_norm=1, amp_norm=1,
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

    def plot_seq_phase(self, t=None, ax=None, show=True, time_norm=1,
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

    def plot_phase_and_amp(self, delta_scale=1, show=True, **kwargs):
        res = self.get_phase_and_amp(**kwargs)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].plot(res['delta'] / delta_scale,
                   res['amplitude']['R'], label='R')
        ax[0].legend()
        ax[0].set_ylabel('amplitude')

        ax[1].plot(res['delta'] / delta_scale,
                   res['phase']['R'] / np.pi, label='R')
        ax[1].plot(res['delta'] / delta_scale,
                   res['phase']['T'] / np.pi, label='T')
        ax[1].set_ylabel('phase (units of pi)')
        ax[1].legend()

        for cax in ax:
            cax.grid()
            cax.set_xlabel('detuning')
            cax.set_xlim(res['delta'].min() / delta_scale,
                         res['delta'].max() / delta_scale)
        if show:
            plt.show()
            return
        else:
            return fig, ax


# %% Tests

if __name__ == '__main__':

    # -- sequence plotting test
    if False:
        rect_pulse = PulseShape(pulse_type='rect', time_offset=0)
        rect_pulse.laser_phase = pi/4

        seq = PulseSequence()
        seq.add_pulse(rect_pulse)
        seq.add_pulse(pulse_type='rect', time_offset=1.5*pi, pulse_duration=pi)
        seq.add_pulse(pulse_type='rect', time_offset=2.5*pi,
                      pulse_duration=pi, window='hanning', laser_phase=pi/6)
        seq.plot_sequence()

    # -- propagator / diffraction test
    if False:
        seq = PulseSequence()
        seq.add_pulse(pulse_type='rect', pulse_duration=0.5*pi)
        seq.global_detuning = 0.5
        U = seq.propagator()
        print(U)
        D = seq.diffraction_matrix()
        print(D)

    # -- get phase and amp test
    if False:
        seq = PulseSequence()
        seq.add_pulse(pulse_type='rect',
                      pulse_duration=0.5*pi,
                      detuning=0)
        delta = np.linspace(-5, 5, 100)
        res = seq.get_phase_and_amp(delta=delta, nodyn=True)
        # - plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].plot(res['delta'], res['amplitude']['R'], label='R')
        ax[0].plot(res['delta'], res['amplitude']['T'], label='T')
        ax[0].legend()

        ax[1].plot(res['delta'], res['phase']['R'] / np.pi, label='R')
        ax[1].plot(res['delta'], res['phase']['T'] / np.pi, label='T')
        for cax in ax:
            cax.grid()

        plt.show()

    # -- plot phase and amp test
    if True:
        seq = PulseSequence()
        seq.add_pulse(pulse_type='rect',
                      pulse_duration=0.5*pi,
                      detuning=0)
        delta = np.linspace(-5, 5, 100)
        res = seq.plot_phase_and_amp(delta=delta, nodyn=True)

    # -- time evolution
    if False:
        seq = PulseSequence(global_detuning=0,
                            decoherence_rate=1e-20)
        seq.add_pulse(pulse_type='rect',
                      pulse_duration=5*pi)
        T = np.linspace(0, 5 * pi, 500)
        res = seq.time_evolution(T)
        # - plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].plot(res['time'], res['amplitude']['01'], label='R')
        ax[0].plot(res['time'], res['amplitude']['00'], label='T')
        ax[0].legend()

        ax[1].plot(res['time'], res['phase']['01'] / np.pi, label='R')
        ax[1].plot(res['time'], res['phase']['00'] / np.pi, label='T')
        for cax in ax:
            cax.grid()

        plt.show()

    # -- Sinc test !
    if False:
        # - initialize
        seq = PulseSequence()
        sinc_minima = 8
        seq.add_pulse(pulse_type='sinc',
                      pulse_duration=0.5*pi,
                      sinc_minima=sinc_minima,
                      rabi_pulsation=sinc_minima*2,
                      window='hanning')

        # seq.plot_sequence()
        # - scan detuning
        delta = np.linspace(-50, 50, 100)
        res = seq.get_phase_and_amp(delta=delta, nodyn=True,
                                    ode_options={'nsteps': 1000})

        # - plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].plot(res['delta'], res['amplitude']['R'], label='R')
        ax[0].plot(res['delta'], res['amplitude']['T'], label='T')
        ax[0].legend()

        ax[1].plot(res['delta'], res['phase']['R'] / np.pi * 180, label='R')
        ax[1].plot(res['delta'], res['phase']['T'] / np.pi * 180, label='T')
        for cax in ax:
            cax.grid()

        plt.show()

    # -- SSinc test !
    if False:
        # - initialize
        seq = PulseSequence()
        sinc_minima = 10
        seq.add_pulse(pulse_type='ssinc',
                      phi_sinc=10 * np.pi / 180,
                      pulse_duration=0.5*pi,
                      sinc_minima=sinc_minima,
                      rabi_pulsation=sinc_minima*2,
                      window='blackman')

        # seq.plot_sequence()
        # - scan detuning
        delta = np.linspace(-50, 50, 100)
        res = seq.get_phase_and_amp(delta=delta, nodyn=True)

        # - plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].plot(res['delta'], res['amplitude']['R'], label='R')
        ax[0].plot(res['delta'], res['amplitude']['T'], label='T')
        ax[0].set_ylabel('amplitude')
        ax[0].legend()

        ax[1].plot(res['delta'], res['phase']['R'] / np.pi * 180, label='R')
        ax[1].plot(res['delta'], res['phase']['T'] / np.pi * 180, label='T')
        ax[1].set_ylabel('phase (deg.)')
        for cax in ax:
            cax.grid()
            cax.set_xlabel('detuning')

        plt.show()


