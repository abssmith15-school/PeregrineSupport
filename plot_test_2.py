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

climit = [-120,-30]
title = input['title'] if 'title' in input else '2D Childs Bay Slice [330 deg]'
tenlog = 1
what_to_plot = None
how_to_collapse_over_frequency = 'incoherent_mean'
raw = False
rlimit = None
zlimit = None
write_png_instead = False

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'title': title = value
    if key == 'climit': climit = [float(x) for x in value.split(',', 1)]
    if key == 'rlimit' or key == 'xlimit': rlimit = [float(x) for x in value.split(',', 1)]
    if key == 'zlimit' or key == 'ylimit': zlimit = [float(x) for x in value.split(',', 1)]
    if key == 'what_to_plot': what_to_plot = value
    if key == 'tenlog': tenlog = int(value)
    if key == 'collapse_over_frequency': how_to_collapse_over_frequency = value
    if key == 'png': write_png_instead = True
    if key == 'raw': raw = True

# choose an obvious default thing to plot if none specified
if not what_to_plot:
    if 'power' in input: what_to_plot = 'power'
    elif 'pressure' in input: what_to_plot = 'pressure'
    elif 'env' in input: what_to_plot = 'env'
    else: raise RuntimeError('need to specify what_to_plot')

# mmap the desired data segment of the desired file as a read-only numpy array
value = numpy.memmap(
    input[what_to_plot]['path'],
    mode='r',
    offset=input[what_to_plot].get('offset',
           input[what_to_plot]['data_offset_in_file']),
    dtype=numpy.dtype(input[what_to_plot]['dtype']),
    shape=tuple(input[what_to_plot]['shape'])
)
print(
    'mmap info:',
    'dtype =', value.dtype,
    'shape =', value.shape,
    'min/max =', numpy.nanmin(value), numpy.nanmax(value),
    file=sys.stderr
)
print(
    'column energy (first 10):',
    [numpy.nanmax(value[i, :]) for i in range(10)],
    file=sys.stderr
)

dimnames = input[what_to_plot]['dimnames']

# collapse complex outputs
if numpy.iscomplexobj(value):
    if 'F' in dimnames and how_to_collapse_over_frequency != 'incoherent_mean':
        idim = dimnames.index('F')
        F = value.shape[idim]

        if 'hann_then_mean' == how_to_collapse_over_frequency:
            window = 2.0 * numpy.hanning(F + 2)[1:(F + 1)] * F / (F + 1.0)
            value = numpy.mean(value * window, idim)
        elif 'keep_first' == how_to_collapse_over_frequency:
            value = numpy.take(value, 0, axis=idim)
        else:
            value = numpy.mean(value, idim)

        dimnames.remove('F')

    value = value.real * value.real + value.imag * value.imag

# collapse excess dimensions
for label in ['F', 'G', 'V', 'S', 'B']:
    if label in dimnames:
        idim = dimnames.index(label)
        if 'F' == label and 'keep_first' == how_to_collapse_over_frequency:
            value = numpy.take(value, 0, axis=idim)
        else:
            value = numpy.sum(value, idim) if 'V' == label else numpy.mean(value, idim)
        dimnames.remove(label)

if (not 'R' in dimnames) or (not 'Z' in dimnames) or dimnames.index('R') > dimnames.index('Z') or len(value.shape) != 2:
    raise RuntimeError('input is not R by Z')

# extents
rextent = numpy.asarray(
    input[what_to_plot]['range_limits'] if 'range_limits' in input[what_to_plot]
    else input['range_limits'],
    dtype=float
)

zextent = numpy.asarray(input['depth_limits'], dtype=float) * -1

# downsample in R
if not raw and value.shape[0] >= 800:
    downsample = value.shape[0] // 400

    if value.shape[0] % downsample:
        rextent[1] -= (value.shape[0] % downsample) * (rextent[1] - rextent[0]) / value.shape[0]
        value = value[0:(value.shape[0] - (value.shape[0] % downsample)), :]

    value.shape = [value.shape[0] // downsample, downsample, value.shape[1]]
    value = numpy.mean(value, axis=1)

# downsample in Z
downsample = value.shape[1] // ((value.shape[0] * 3) // 5)
if not raw and downsample > 1:
    if value.shape[1] % downsample:
        zextent[1] -= (value.shape[1] % downsample) * (zextent[1] - zextent[0]) / value.shape[1]
        value = value[:, 0:(value.shape[1] - (value.shape[1] % downsample))]

    value.shape = [value.shape[0], value.shape[1] // downsample, downsample]
    value = numpy.mean(value, axis=2)

[R, Z] = value.shape

# units
if zextent[0] > 0 and zextent[1] < 0:
    zextent[0] *= -1
    zextent[1] *= -1

if rextent[1] >= 3e3:
    rextent[0] /= 1e3
    rextent[1] /= 1e3
    r_unit = 'km'
    if rlimit is not None:
        rlimit[0] /= 1e3
        rlimit[1] /= 1e3
else:
    r_unit = 'm'

if abs(zextent[1]) >= 3e3:
    zextent[0] /= 1e3
    zextent[1] /= 1e3
    z_unit = 'km'
    if zlimit is not None:
        zlimit[0] /= 1e3
        zlimit[1] /= 1e3
else:
    z_unit = 'm'
# mmap bathymetry slice
bathy = None
if 'bathymetry' in input:
    bathy = numpy.memmap(
        input['bathymetry']['path'],
        mode='r',
        offset=input['bathymetry'].get('offset',
               input['bathymetry']['data_offset_in_file']),
        dtype=numpy.dtype(input['bathymetry']['dtype']),
        shape=tuple(input['bathymetry']['shape'])
    ).astype(float)
    bathy = -bathy  # positive depth

    r_bathy = numpy.linspace(rextent[0], rextent[1], bathy.shape[0])


# plotting
dpi = 300 if not raw else max(R / 6.4, Z / 4.8)
fig, ax = matplotlib.pyplot.subplots(dpi=dpi)

if tenlog:
    value = 10.0 * numpy.log10(value + 2e-38)
    print(
    'TL range (dB):',
    numpy.nanmin(value),
    numpy.nanmax(value),
    file=sys.stderr
)
if climit[0] < 0 and climit[1] < 0:
    cmap = 'turbo_r'
    vmin =-climit[1]
    vmax =-climit[0]
else:  
    cmap = 'turbo'
    vmin = climit[0]
    vmax = climit[1]

pos = ax.imshow(
    numpy.transpose(value),
    extent=[rextent[0], rextent[1], zextent[0], zextent[1]],
    origin='lower',
    aspect='auto',
    cmap=cmap,
    vmin=vmin,
    vmax=vmax
)

# bathymetry overlay
if bathy is not None:
    ax.plot(r_bathy, bathy, 'k', linewidth=1.5)
    ax.fill_between(r_bathy, bathy, zextent[1], color='k', alpha=0.25)

ax.invert_yaxis()
ax.set(xlabel=f'Range ({r_unit})', ylabel=f'Depth ({z_unit})', title=title)
ax.title.set_size(10)

if rlimit is not None: ax.set_xlim(rlimit)
if zlimit is not None: ax.set_ylim(zlimit)

fig.colorbar(pos, ax=ax, shrink=(Z / R))

if sys.stdout.isatty():
    fig.savefig('/tmp/out.pdf', bbox_inches='tight')
    os.system('open /tmp/out.pdf 2>/dev/null')

fig.savefig(sys.stdout.buffer, bbox_inches='tight')
