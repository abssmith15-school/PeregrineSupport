import sys
import numpy as np
import matplotlib.pyplot as plt
import json

def read_obb_json_plan(filename,variable_name, get_bounds = 0):
    with open(filename,'rb') as f:
    # consume json on input until eof, and parse into a python dict
        input = json.load(f)
    # mmap the data segment of the bathy.obb file into a read-only numpy array
        obj = np.memmap(input[f'{variable_name}']['path'], mode = 'r',
                            offset = input[f'{variable_name}']['data_offset_in_file'],
                            dtype = np.dtype(input[f'{variable_name}']['dtype']),
                            shape = tuple(input[f'{variable_name}']['shape']))

        #range_limits = np.array(input["range_limits"])
        #depth_limits = np.array(input["depth_limits"])
    if get_bounds == 1:
        boundsWESN = np.array((input['bounds']['west'],input['bounds']['east'],input['bounds']['south'],input['bounds']['north']))
        return obj,boundsWESN

    return obj

#def read_obb_json_slice(filename,variable_name, get_bounds = 0):
 #   with open(filename,'rb') as f:
    # consume json on input until eof, and parse into a python dict
  #      input = json.load(f)
   #     breakpoint()
    # mmap the data segment of the bathy.obb file into a read-only numpy array
    #    obj = np.memmap(input[f'{variable_name}']['path'], mode = 'r',
     #                       offset = input[f'{variable_name}']['data_offset_in_file'],
      #                      dtype = np.dtype(input[f'{variable_name}']['dtype']),
       #                     shape = tuple(input[f'{variable_name}']['shape']))

        #range_limits = np.array(input["range_limits"])
        #depth_limits = np.array(input["depth_limits"])
   # if get_bounds == 1:
    #    boundsRngDepth = np.array((input['range_limits'],input['depth_limits']))
        #"bounds": { "west": 36.07317, "east": 39.75017, "south": -48.03238, "north": -45.51428 },
	    #"relative_bounds": { "west": -140012.8, "east": 140012.8, "south": -140000, "north": 140000 }
     #   return obj,boundsRngDepth

#    return obj

def plot_bathy(X,Y,bathy):
    shallow_levels = np.arange(-500, 500, 40)     # -500 m to 0 m, 20 m spacing
    deep_levels    = np.arange(-6000, -500, 200) # deep ocean, 200 m spacing
    levels = np.unique(np.concatenate((deep_levels, shallow_levels)))
    contours = plt.contour(X, Y, bathy, levels=levels, colors='black', linewidths=0.6)
    plt.clabel(contours, inline=True, fontsize=7, fmt='%d')
    plt.pcolormesh(X,Y,bathy,shading = 'auto')
    plt.colorbar()  
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.title('Bathymetry Grid (Meters)')
    plt.show()

    
def plot_ssp(boundsRngDepth, env, bathy_slice, n_plots=5):

    depths = np.linspace(boundsRngDepth[1][0],
                         boundsRngDepth[1][1]*-1,
                         env.shape[1])

    ranges = np.linspace(boundsRngDepth[0][0],
                         boundsRngDepth[0][1],
                         env.shape[0])

    step = env.shape[0] // n_plots

    plt.figure()

    for i in range(0, env.shape[0], step):

        ssp = env[i, :, 0]
        water_depth = -bathy_slice[i]   # make positive down

        # Find indices where depth is above seabed
        valid = depths <= water_depth

        ssp_water = ssp[valid]
        depths_water = depths[valid]

        print(f"Range {ranges[i]/1000:.2f} km")
        print(f"   Water depth: {water_depth:.1f} m")
        print(f"   Min SSP (water only): {ssp_water.min():.2f}")
        print(f"   Max SSP (water only): {ssp_water.max():.2f}")

        plt.plot(ssp_water, depths_water,
                 label=f'{ranges[i]/1000:.1f} km')

    plt.gca().invert_yaxis()
    plt.xlabel('Sound Speed (m/s)')
    plt.ylabel('Depth (m)')
    plt.title('Water Column SSP Along Slice')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    file_name_pv = '/home/abiga/results/environment_Aug_RL_phi2.json'
    file_name_slice = '/home/abiga/results/environment_bb_slice_340.json'

    #bathy,boundsWESN = read_obb_json_plan(file_name,'bathymetry',1)
    bathy = read_obb_json_plan(file_name_pv,'bathymetry')
   # bathy_slice = read_obb_json_slice(file_name_slice, 'bathymetry')
    #power= read_obb(file_name,'power')
    env_plan,boundsWESN = read_obb_json_plan(file_name_pv,'env',1)
   # env_slice,boundsRngDepth = read_obb_json_slice(file_name_slice,'env',1)

    x = np.linspace(boundsWESN[0],boundsWESN[1],bathy.shape[1])
    y = np.linspace(boundsWESN[2],boundsWESN[3],bathy.shape[0])
    X,Y = np.meshgrid(x,y)
    plot_bathy(X,Y,bathy)
   # print("env_slice shape:", env_slice.shape)
   # print("bathy_slice shape:", bathy_slice.shape)

   # plot_ssp(boundsRngDepth,env_slice,bathy_slice,n_plots=5)
   # breakpoint()
    
   
    #plot_ssp(range,depth,SSP)

    #How to plot SSP from env grid: The three channels of the C dimension are sound speed, density, and attenuation
   #SSP grid: env[:,:,:,0]
