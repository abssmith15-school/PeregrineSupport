clear
close all

%% ------------------------------------------------------------------------
% INPUTS
%% ------------------------------------------------------------------------

ncfile = 'C:\Users\abiga\Downloads\GEBCO_01_Jul_2026_9709b50e0487\gebco_2026_n41.352_s40.9922_w9.5073_e9.985.nc';

lat0 = 41.1719;
lon0 = 9.7459;

btyfile = 'gebco20km.bty';
interp_type = 'RS';

%% ------------------------------------------------------------------------
% READ GEBCO
%% ------------------------------------------------------------------------

lon = ncread(ncfile,'lon');
lat = ncread(ncfile,'lat');
elev = ncread(ncfile,'elevation');

% GEBCO stores elevation as
% ocean = negative
% land  = positive

%% ------------------------------------------------------------------------
% Convert to Bellhop bathymetry
%% ------------------------------------------------------------------------

depth = -double(elev);

% Any land becomes zero depth
depth(depth<0)=0;

%% ------------------------------------------------------------------------
% Convert lat/lon to local x-y coordinates (km)
%% ------------------------------------------------------------------------

Re = 6371000;

deg2rad = pi/180;

x = (lon-lon0) * deg2rad * Re * cos(lat0*deg2rad)/1000;
y = (lat-lat0) * deg2rad * Re /1000;

%% ------------------------------------------------------------------------
% Make sure dimensions agree
%% ------------------------------------------------------------------------

% GEBCO is typically elevation(lon,lat)

if size(depth,1)==length(lon) && size(depth,2)==length(lat)
    depth = depth';
end

%% ------------------------------------------------------------------------
% Build Bathy structure
%% ------------------------------------------------------------------------

Bathy.X = x;
Bathy.Y = y;
Bathy.depth = depth;

%% ------------------------------------------------------------------------
% Write Bellhop3D bathymetry
%% ------------------------------------------------------------------------

writebdry3d(btyfile,interp_type,Bathy);

fprintf('Wrote %s\n',btyfile);

%% ------------------------------------------------------------------------
% Plot check
%% ------------------------------------------------------------------------

figure
imagesc(x,y,depth)
axis xy
axis equal
xlabel('X (km)')
ylabel('Y (km)')
title('Bellhop Bathymetry')
colorbar