import os
import sys
import json
from scipy.io import savemat

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

climit = [-140, -60]
title = input['title'] if 'title' in input else '2D Namaqua Fossil Forest Slice [340 deg]'
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
    else: raise RuntimeError('need to specify what_to_plot')

# mmap the desired data segment of the desired file as a read-only numpy array with the right shape and data type
value = numpy.memmap(
    input[what_to_plot]['path'],
    mode='r',
    offset=input[what_to_plot]['offset'] if 'offset' in input[what_to_plot]
           else input[what_to_plot]['data_offset_in_file'],
    dtype=numpy.dtype(input[what_to_plot]['dtype']),
    shape=tuple(input[what_to_plot]['shape'])
)

dimnames = input[what_to_plot]['dimnames']
   # --- DEBUG: inspect bathymetry ---
if 'bathymetry' in input:
    bathy = numpy.memmap(
        input['bathymetry']['path'],
        mode='r',
        offset=input['bathymetry'].get(
            'offset',
            input['bathymetry']['data_offset_in_file']
        ),
        dtype=numpy.dtype(input['bathymetry']['dtype']),
        shape=tuple(input['bathymetry']['shape'])
    )

    print(
        'BATHY DEBUG:',
        'dtype =', bathy.dtype,
        'shape =', bathy.shape,
        'min/max =', numpy.nanmin(bathy), numpy.nanmax(bathy),
        file=sys.stderr
    )

    print(
        'BATHY first 10 values:',
        bathy[:10],
        file=sys.stderr
    )

if numpy.iscomplexobj(value):
    if 'F' in dimnames and how_to_collapse_over_frequency != 'incoherent_mean':
        # handle frequency dimension specially for complex outputs
        idim = dimnames.index('F')
        F = value.shape[idim]
        if 'hann_then_mean' == how_to_collapse_over_frequency:
            # this matches what peregrine does internally but both need to be verified
            window = 2.0 * numpy.hanning(F + 2)[1:(F + 1)] * F / (F + 1.0)
            value = numpy.mean(value * window, idim)
        elif 'keep_first' == how_to_collapse_over_frequency:
            value = numpy.take(value, 0, axis=idim)
        else:
            value = numpy.mean(value, idim)

        dimnames.remove('F')
        
    # is there a more efficient way to get the magnitude-squared of a complex numpy array?
    	# --- SAVE COMPLEX PRESSURE BEFORE MAGNITUDE ---        
psi_complex = value.copy()   # [R x Z] or [Z x R] depending on dim order
value = value.real * value.real + value.imag * value.imag

# if any of these possible "excess dimensions" are present, just plot the mean or sum over them
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

rextent = numpy.asarray(
    input[what_to_plot]['range_limits'] if 'range_limits' in input[what_to_plot]
    else input['range_limits'],
    dtype=float
)

zextent = numpy.asarray(input['depth_limits'], dtype=float) * -1

if not raw and value.shape[0] >= 800:
    downsample = value.shape[0] // 400
    print('downsampling in R from %u to %u' % (value.shape[0], value.shape[0] // downsample), file=sys.stderr)

    # discard remainder in range if necessary
    if value.shape[0] % downsample:
        rextent[1] -= (value.shape[0] % downsample) * (rextent[1] - rextent[0]) / value.shape[0]
        value = value[0:(value.shape[0] - (value.shape[0] % downsample)), :]

    # split the dimension to be downsampled
    value.shape = [value.shape[0] // downsample, downsample, value.shape[1]]
    value = numpy.mean(value, axis=1)

downsample = value.shape[1] // ((value.shape[0] * 3) // 5)
if not raw and downsample > 1:
    print('downsampling in Z from %u to %u' % (value.shape[1], value.shape[1] // downsample), file=sys.stderr)

    if value.shape[1] % downsample:
        zextent[1] -= (value.shape[1] % downsample) * (zextent[1] - zextent[0]) / value.shape[1]
        value = value[:, 0:(value.shape[1] - (value.shape[1] % downsample))]

    value.shape = [value.shape[0], value.shape[1] // downsample, downsample]
    value = numpy.mean(value, axis=2)

[R, Z] = value.shape;
# --- SELECT RECEIVER RANGE INDEX ---
receiver_km = 200.0  # <-- CHANGE THIS
rvec = numpy.linspace(rextent[0], rextent[1], R)
irx = numpy.argmin(numpy.abs(rvec - receiver_km))
# --- COMPLEX PRESSURE AT RECEIVER (z-dependent) ---
psi_z = psi_complex[irx, :]    # shape: [Z], complex
# --- SAVE COMPLEX PRESSURE FOR TIME-DOMAIN SYNTHESIS ---
savemat(
    "peregrine_psi_complex.mat",
    {
        "psi_z": psi_z,                         # complex [Z]
        "depth": numpy.linspace(zextent[0], zextent[1], Z),
        "range_km": receiver_km,
        "title": title
    }
)

print("Saved peregrine_psi_complex.mat", file=sys.stderr)

z_unit = 'm'
# if negative-is-down zextents were given, flip their signs
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

    z_unit = 'm'

dpi = 300 if not raw else max(R / 6.4, Z / 4.8)

# plot it and save to pdf
fig, ax = matplotlib.pyplot.subplots(dpi=dpi)

if tenlog:
    value = 10.0 * numpy.log10(value + 2e-38)

if climit[0] < 0 and climit[1] < 0:
    cmap = 'turbo_r'
    vmin = -climit[1]
    vmax = -climit[0]
    value = -1.0 * value
    # --- SAVE EXACTLY-PLOTTED FIELD TO MATLAB ---
    mat_dict = {
    "TL_dB": value,                         # EXACT plotted field [R x Z]
    "range": numpy.linspace(rextent[0], rextent[1], value.shape[0]),
    "depth": numpy.linspace(zextent[0], zextent[1], value.shape[1]),
    "climit": climit,
    "title": title,
    "r_unit": r_unit,
    "z_unit": z_unit} 
    savemat("peregrine_TL_plotted.mat", mat_dict)
    print("Saved peregrine_TL_plotted.mat", file=sys.stderr)

else:
    cmap = 'turbo'
    vmin = climit[0]
    vmax = climit[1]

if write_png_instead:
    matplotlib.pyplot.imsave(
        '/tmp/out.png' if sys.stdout.isatty() else sys.stdout,
        numpy.transpose(value),
        cmap=cmap, vmin=vmin, vmax=vmax, origin='upper'
    )
    if sys.stdout.isatty():
        os.system('open /tmp/out.png')
else:
    pos = ax.imshow(
        numpy.transpose(value),
        extent=[rextent[0], rextent[1], zextent[0], zextent[1]],
        origin='lower',
        aspect='auto',
#(((rextent[1] - rextent[0]) * Z) / ((zextent[1] - zextent[0]) * R)),
        cmap=cmap, vmin=vmin, vmax=vmax
     ) 
  # --- Correct bathymetry overlay ---
if 'bathymetry' in input:
    bathy = -bathy  # convert to positive depth

    r = numpy.linspace(rextent[0], rextent[1], bathy.shape[0])

    ax.plot(
        r,
        bathy,
        'k',
        linewidth=1.5,
        zorder=10
    )
    z_unit = 'm'
# --- Bathymetry overlay ---

   # ax.fill_between(r,z_bathy, zmax, color = 'k')
   # ax.set_ylim(0,zmax)


   # if rlimit is not None: ax.set_xlim(rlimit)
   # if zlimit is not None: ax.set_ylim(zlimit)


    # draw bathymetry line only
    # --- Solid bottom / land fill ---

z_surface = 0.0               # water surface
z_bottom = zextent[1]         # max plot depth (positive)

# fill solid earth BELOW bathymetry
ax.fill_between(
    r,
    bathy,
    z_bottom,
    color='k',
    alpha=0.25,
    zorder=5
)

# fill land ABOVE water surface (where bathy < 0)
ax.fill_between(
    r,
    z_surface,
    bathy,
    where=(bathy < z_surface),
    color='k',
    alpha=0.4,
    zorder=5
)
ax.invert_yaxis()
ax.set(xlabel='Range (%s)' % r_unit)
ax.set(ylabel='Depth (%s)' % z_unit)
ax.set(title=title)
ax.title.set_size(10)
fig.colorbar(pos, ax=ax, shrink=(Z / R))

if sys.stdout.isatty():
    fig.savefig('/tmp/out.pdf', bbox_inches='tight')
    os.system('open /tmp/out.pdf 2>/dev/null')
    fig.savefig(sys.stdout.buffer, bbox_inches='tight')
