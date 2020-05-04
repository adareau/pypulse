# -*- coding: utf-8 -*-
'''
Author   : alex
Created  : 2020-05-04 14:36:08
Modified : 2020-05-04 16:06:36

Comments :
'''
from sequence import PulseSequence
from shapes import PulseShape
from numpy import pi

rect_pulse = PulseShape(pulse_type='rect', time_offset=0, window='hanning')
rect_pulse.laser_phase = pi/4

seq = PulseSequence()
seq.add_pulse(rect_pulse)
seq.add_pulse(pulse_type='rect', time_offset=pi, pulse_duration=pi)
U = seq.propagator()
print(U)

# -- plot ?
rect_pulse.plot_pulse(time_norm=pi)
seq.plot_sequence()