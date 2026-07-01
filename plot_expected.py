#!/usr/bin/env python3
# the only prerequisite on a vanilla mac with xcode installed is "pip3 install matplotlib", w/o sudo

import os
import sys
import math
import json
import numpy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

xlimit = None
ylimit = None
climit = [0,110]
xtick = 0
ytick = 0
what_to_plot = 'power'
tenlog = 1
title = None
xlabel = 'Distance (%s)'
ylabel = 'Distance (%s)'
how_to_collapse_over_frequency = 'mean'

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'title': title = value
    if key == 'climit': climit = [float(x) for x in value.split(',', 1)]
    if key == 'xlimit': xlimit = [float(x) for x in value.split(',', 1)]
    if key == 'ylimit':ylimit = [float(x) for x in value.split(',', 1)]
    if key == 'xtick': xtick = float(value)
    if key == 'ytick': ytick = float(value)
    if key == 'what_to_plot': what_to_plot = value
    if key == 'tenlog': tenlog = int(value)
    if key == 'collapse_over_frequency': how_to_collapse_over_frequency = value

if not title: title =  ' October TL, Source = 75 m, Reciever = 30 m, Phi = 2 '

# consume json on input until eof, and parse into a python dict
input = json.loads(sys.stdin.read())
print("\n=== TOP LEVEL KEYS ===")
for k in input.keys():
    print(k)
print("======================\n")
print("\n=== POWER FIELD INFO ===")

for k, v in input['power'].items():
    print(k, ":", v)
print("========================\n")
# mmap the data segment of the value.obb file into a read-only numpy array
value = numpy.memmap(input[what_to_plot]['path'], mode = 'r',
                     offset = input[what_to_plot]['data_offset_in_file'],
                     dtype = numpy.dtype(input[what_to_plot]['dtype']),
                     shape = tuple(input[what_to_plot]['shape']))

dimnames = input[what_to_plot]['dimnames']

# if any of these possible "excess dimensions" are present...
for label in ['F', 'G', 'Z', 'V', 'S']:
    if label in dimnames:
        idim = dimnames.index(label)
        if 'F' == label and 'keep_first' == how_to_collapse_over_frequency:
            value = numpy.take(value, 0, axis=idim)
        else:
            value = numpy.sum(value, idim) if 'V' == label else numpy.mean(value, idim)
        dimnames.remove(label)

if (not 'X' in dimnames) or (not 'Y' in dimnames) or dimnames.index('Y') > dimnames.index('X') or len(value.shape) != 2:
    raise RuntimeError('input is not Y by X')

[Y, X] = value.shape

if 'relative_bounds' in input and (input['bounds']['north'] - input['bounds']['south'] < 20.0):
    xextent = numpy.asarray([input['relative_bounds']['west'], input['relative_bounds']['east']], dtype=float)
    yextent = numpy.asarray([input['relative_bounds']['south'], input['relative_bounds']['north']], dtype=float)

    if xextent[1] >= 2e3: xextent[0] /= 1e3; xextent[1] /= 1e3; x_unit = 'km'
    else: x_unit = 'm'

    if yextent[1] >= 2e3: yextent[0] /= 1e3; yextent[1] /= 1e3; y_unit = 'km'
    else: y_unit = 'm'
else:
    xextent = numpy.asarray([input['bounds']['west'], input['bounds']['east']], dtype=float)
    yextent = numpy.asarray([input['bounds']['south'], input['bounds']['north']], dtype=float)
    xlabel = 'Longitude (%s)'
    ylabel = 'Latitude (%s)'
    x_unit = 'deg'
    y_unit = 'deg'

fig, ax = matplotlib.pyplot.subplots(dpi=300)

# Explicit Transmission Loss definition
TL = -10.0 * numpy.log10(value + 1e-18)
#value = TL
#value  = TL
value = np.flipud(value)
value = 188-TL-73
# ------------------------------------------------------------
# Maximum propagation range statistics
# ------------------------------------------------------------

mask = value > 0

if np.any(mask):

    xcoords = np.linspace(xextent[0], xextent[1], X)
    ycoords = np.linspace(yextent[0], yextent[1], Y)

    XX, YY = np.meshgrid(xcoords, ycoords)

    # Source at center of domain
    xc = 0.5 * (xextent[0] + xextent[1])
    yc = 0.5 * (yextent[0] + yextent[1])

    # --------------------------------------------------------
    # Compute radial distance
    # --------------------------------------------------------

    if x_unit == 'deg':

        # Convert geographic degrees to km
        lat_km = 111.32
        lon_km = 111.32 * np.cos(np.deg2rad(yc))

        dx = (XX - xc) * lon_km
        dy = (YY - yc) * lat_km

    else:

        # Already in meters or km
        dx = XX - xc
        dy = YY - yc

        if x_unit == 'm':
            dx /= 1000.0
            dy /= 1000.0

    R = np.sqrt(dx**2 + dy**2)

    # --------------------------------------------------------
    # Bearing angle
    # 0 deg = north
    # 90 deg = east
    # --------------------------------------------------------

    theta = np.degrees(np.arctan2(dx, dy))
    theta[theta < 0] += 360

    # --------------------------------------------------------
    # Sector masks
    # --------------------------------------------------------

    sectorA = mask & (theta >= 165) & (theta <= 330)
    sectorB = mask & ((theta < 165) | (theta > 330))

    # --------------------------------------------------------
    # Maximum ranges
    # --------------------------------------------------------

    maxA = np.max(R[sectorA]) if np.any(sectorA) else 0
    maxB = np.max(R[sectorB]) if np.any(sectorB) else 0
    avgRangeA = np.mean(R[sectorA]) if np.any(sectorA) else float('nan')
    avgRangeB = np.mean(R[sectorB]) if np.any(sectorB) else float('nan')
    print("\n===================================")
    print(f"160-330 deg sector max range : {maxA:.2f} km")
    print(f"Other sector max range       : {maxB:.2f} km")
    print("===================================\n")
    print(f"160-330 deg sector avg range : {avgRangeA:.2f} km")
    print(f"Other sector avg range       : {avgRangeB:.2f} km")
else:
    print("No locations found where value > 0")
#cmap = 'viridis'
from matplotlib.colors import BoundaryNorm

vmin = 0
vmax = 60
cmap = ('turbo_r')


pos = ax.imshow(value,
    extent=[xextent[0], xextent[1], yextent[0], yextent[1]],
    aspect=(((xextent[1] - xextent[0]) * Y) / ((yextent[1] - yextent[0]) * X)),
    cmap=cmap, vmin=0,
    vmax=60)

if xlimit: ax.set_xlim(xlimit)
if ylimit: ax.set_ylim(ylimit)

# we want north at the top


if (xtick): ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xtick));
if (ytick): ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ytick));

ax.set_xlabel(xlabel % x_unit, fontsize=16)
ax.set_ylabel(ylabel % y_unit, fontsize=16)
ax.set_title(title, fontsize=16)

# Tick label size (VERY important)
ax.tick_params(axis='both', labelsize=14)


cbar = fig.colorbar(pos, ax=ax, shrink=0.6)

# Colorbar label (the title on the bar)
cbar.set_label("Transmission Loss dB", fontsize=16)

# Tick labels on the colorbar
cbar.ax.tick_params(labelsize=14)

# write pdf to stdout
if (sys.stdout.isatty()):
    fig.savefig('/tmp/out.pdf', bbox_inches='tight')
    os.system('open /tmp/out.pdf 2>/dev/null')
else: fig.savefig(sys.stdout.buffer, bbox_inches='tight')
 
# Save variables needed for plotting to MATLAB .mat file

mat_dict = {
    "value": value,
    "xextent": xextent,
    "yextent": yextent,
    "vmin": vmin,
    "vmax": vmax,
    "X": X,
    "Y": Y
}

savemat("Dec_NLW_Test.mat", mat_dict)

print("Saved debug_plot.mat")
