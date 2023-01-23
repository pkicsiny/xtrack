# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

num_turns = 10
num_particles = 30# 100000 # Enough to saturate a high-end GPU

#################################
# Load a line and build tracker #
#################################

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

tracker = line.build_tracker()

###########################
# Generate some particles #
###########################

particles = tracker.build_particles(
    x_norm=np.linspace(-2, 2, num_particles), y_norm=0.1, delta=3e-4,
    nemitt_x=2.5e-6, nemitt_y=2.5e-6)

####################
# Optimize tracker #
####################

tracker.optimize_for_tracking()

###########################
# Track with optimization #
###########################

tracker.track(particles, num_turns=num_turns, time=True)
tracking_time = tracker.time_last_track

particles.move(_context = xo.ContextCpu())

assert len(particles.state) == num_particles
assert np.all(particles.state == 1)

print(f'Tracking time [us/p/turn]: {tracking_time*1e6/num_particles/num_turns:.2f}')