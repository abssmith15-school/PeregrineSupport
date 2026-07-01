import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Load GEBCO dataset
ds = nc.Dataset("GEBCO_2025_sub_ice.nc")
lon = ds["lon"][:]
lat = ds["lat"][:]
bathy = ds["elevation"][:]

# Create figure with orthographic projection (round globe)
plt.figure(figsize=(8,8))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-160, central_latitude=25))
ax.coastlines()
ax.gridlines()

# Plot bathymetry
ax.pcolormesh(lon, lat, bathy, transform=ccrs.PlateCarree(), cmap="viridis")

plt.title("Bathymetry on a Globe")

# SAVE to PDF
plt.savefig("bathymetry_globe.pdf", bbox_inches="tight")

# Show on screen
