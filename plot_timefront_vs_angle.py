#!/usr/bin/env python3
# usage: /peregrine time 20090101T000000Z lon -145.05218d lat 75.36330d bearing -110d range 175026 rx_depth -100:-10:-400 tx_depth -100 fc 100 bandwidth 19.2 df 0.3 grain_size 8 output_complex 1 title 'Arrivals vs time and angle' | ./plot_timefront_vs_angle.py climit -120,-54 c0_retard 1505
import os
import sys
import json

# consume json on input until eof, and parse into a python dict. we do this prior to
# importing any libraries that are not part of the standard python3 installation, so that
# the upstream process writing to stdout can finish successfully without getting a SIGPIPE
# if importing fails
input = json.load(sys.stdin)

try:
    import numpy as np
    import matplotlib
except:
    print('warning: plotting skipped, please "pip3 install matplotlib" or the equivalent on your OS', file=sys.stderr)
    exit()

matplotlib.use('pdf')
import matplotlib.pyplot

climit = [-90, -30]
title = input['title'] if 'title' in input else 'don\'t forget a plot title'
xlabel = 'Time (s)'
c0_retard = 0
time_domain_offset = None

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'title': title = value
    if key == 'climit': climit = [float(x) for x in value.split(',', 1)]
    if key == 'time_domain_offset': time_domain_offset = float(value)
    if key == 'c0_retard': c0_retard = float(value)

if 'power' in input:
    what_to_plot = 'power'
else:
    what_to_plot = 'pressure'

# mmap the data segment of the file into a read-only numpy array
value = np.memmap(input[what_to_plot]['path'], mode = 'r',
                     offset = input[what_to_plot]['offset'] if 'offset' in input[what_to_plot] else input[what_to_plot]['data_offset_in_file'],
                     dtype = np.dtype(input[what_to_plot]['dtype']),
                     shape = tuple(input[what_to_plot]['shape']))

dimnames = input[what_to_plot]['dimnames']

# if any of these possible "excess dimensions" are present, just keep one
for label in ['G', 'B', 'R']:
    if label in dimnames:
        idim = dimnames.index(label)
        # keep the middle index along the dimension for G and B, the last one for R
        value = np.take(value, value.shape[idim] - 1 if 'R' == label else value.shape[idim] // 2, axis=idim)
        dimnames.remove(label)

idim = dimnames.index('F')
[Z, F] = value.shape[0:2]
frequencies = np.asarray(input['input']['frequencies'] if 'input' in input else input['frequencies'])
omega = 2.0 * np.pi * frequencies
df = frequencies[1] - frequencies[0]

if not time_domain_offset:
    time_domain_offset = 0.125 / df

if c0_retard != 0 and ('range' in input or 'range_limits' in input):
    # FIXME: make this work when number of ranges is greater than 1
    r = np.max(np.asarray(input['range_limits'])) if 'range_limits' in input else np.asarray(input['range'])
    time_shift = time_domain_offset - r / c0_retard
    xlabel = 'Time (s) vs %g m/s' % c0_retard

else: time_shift = time_domain_offset;

shift = np.exp(-1j * omega * time_shift)

# do the time domain inversion
time_window = 2.0 * np.hanning(F + 2)[1:(F + 1)] * F / (F + 1.0)
value = np.fft.ifft(value * time_window * shift, axis=idim)

tlimit = (-time_domain_offset, 1.0 / df - time_domain_offset)
T = F

if 'rx_depths' in input:
    depths = np.asarray(input['rx_depths'])
    dz = np.abs(depths[1] - depths[0])
else:
    depths = np.asarray(input['input']['depth_limits'] if 'input' in input else input['depth_limits'])
    dz = np.abs(depths[1] - depths[0]) / (Z - 1)

Y = Z
# pixel-edge limits of k space, works for even or odd phone count as long as uniformly spaced
klimit = (Y // 2 - np.asarray((0, Y)) + 0.5) / (Y * dz)

# mapping of k space to cosine theta
kfree = 2 * np.pi * frequencies[F // 2] / 1500.0
ylimit = klimit / kfree

angle_window = 2.0 * np.hanning(Z + 2)[1:(Z + 1)] * Z / (Z + 1.0)
value = np.fft.fftshift(np.fft.fft(value * angle_window.reshape([-1, 1]), axis=0), axes=[0])

# and convert to magnitude squared
value = value.real * value.real + value.imag * value.imag

# plot it and save to pdf
fig, ax = matplotlib.pyplot.subplots(dpi=300)

# note the values and the colormap are reversed so that tick marks on the colorbar are positive
pos = ax.imshow(-10.0 * np.log10(value + 1e-30),
    extent=[tlimit[0], tlimit[1], ylimit[0], ylimit[1]],
    aspect=np.abs(((tlimit[1] - tlimit[0]) * Y / ((ylimit[1] - ylimit[0]) * T))),
    cmap='turbo_r', vmin=-climit[1], vmax=-climit[0], origin='lower')

ax.set(xlabel=xlabel)
ax.invert_yaxis()
ax.set(ylabel='cos(elev angle)')
ax.set(title=title)
ax.title.set_size(10)

#fig.colorbar(pos, ax=ax, shrink=min(Z/T, 1))

if (sys.stdout.isatty()):
    fig.savefig('/tmp/out.pdf', bbox_inches='tight')
    os.system('open /tmp/out.pdf 2>/dev/null')
else: fig.savefig(sys.stdout.buffer, bbox_inches='tight')
