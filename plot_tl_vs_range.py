#!/usr/bin/env python3

import os
import sys
import json

# consume json on input until eof, and parse into a python dict. we do this prior to
# importing any libraries that are not part of the standard python3 installation, so that
# the upstream process writing to stdout can finish successfully without getting a SIGPIPE
# if importing fails
input = json.load(sys.stdin)

try:
    import numpy
    import matplotlib
except:
    print('warning: plotting skipped, please "pip3 install matplotlib" or the equivalent on your OS', file=sys.stderr)
    exit()

matplotlib.use('pdf')
import matplotlib.pyplot

what_to_plot = 'power'
tenlog = -1
ylabel = None
ylimit = None
title = input['title'] if 'title' in input else 'don\'t forget a plot title'

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'title': title = value
    if key == 'what_to_plot': what_to_plot = value
    if key == 'tenlog': tenlog = int(value)
    if key == 'ylabel': ylabel = value
    if key == 'ylimit': ylimit = [float(x) for x in value.split(',', 1)]

if -1 == tenlog:
    if what_to_plot == 'power': tenlog = 1
    else: tenlog = 0

if not ylabel and what_to_plot == 'power' and tenlog == 1:
    ylabel = 'Propagation loss (dB)'
else:
    ylabel = what_to_plot

# mmap the data segment of the power.obb file into a read-only numpy array
value = numpy.memmap(input[what_to_plot]['path'], mode = 'r',
                     offset = input[what_to_plot]['offset'] if 'offset' in input[what_to_plot] else input[what_to_plot]['data_offset_in_file'],
                     dtype = numpy.dtype(input[what_to_plot]['dtype']),
                     shape = tuple(input[what_to_plot]['shape']))

# reconstruct vector of ranges for plotting tl vs range
R = value.shape[0]
rlimit = input['range_limits']
dr = (rlimit[1] - rlimit[0]) / R
ranges = numpy.arange(rlimit[0] + dr/2, rlimit[1] + dr/2, dr)

if rlimit[1] >= 3e3: ranges /= 1e3; r_unit = 'km'
else: r_unit = 'm'

fig, ax = matplotlib.pyplot.subplots(dpi=300)

if tenlog:
    value = -10.0 * numpy.log10(value + 1e-18)
    ax.invert_yaxis()

pos = ax.plot(ranges, value)

ax.set(xlabel='Range (%s)' % r_unit)
ax.set(ylabel=ylabel)
ax.set(title=title)
ax.title.set_size(10)

if ylimit: ax.set_ylim(ymin=ylimit[0], ymax=ylimit[1])

ax.grid(linewidth=0.5)

if (sys.stdout.isatty()):
    fig.savefig('/tmp/out.pdf', bbox_inches='tight')
    os.system('open /tmp/out.pdf 2>/dev/null')
else: fig.savefig(sys.stdout.buffer, bbox_inches='tight')
