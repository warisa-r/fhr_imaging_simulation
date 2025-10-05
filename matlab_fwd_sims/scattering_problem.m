%% Define constants
light_speed = 299792458;
frequency = 5e9;
k = 2 * pi * frequency / light_speed;
wavelength = light_speed / frequency;
mesh_size = wavelength / 6; % Use a slightly finer mesh

%% Create a geometry
model = createpde;

% Create outer square
outerSquare = [3 4 0 1 1 0 0 0 1 1]';

%% Wavy obstacle parameters
width  = 0.4;       % Width of the inner rectangular hole
height = 0.2;       % Height of the inner rectangular hole
xCenter = 0.5;
yCenter = 0.5;

xLeft   = xCenter - width/2;
xRight  = xCenter + width/2;
yBottom = yCenter - height/2;
yTop    = yCenter + height/2;

NwavePts = 100;      % Number of wave points for smoothness

%% --- Bottom wave parameters ---
ampBottom = 0.01;
freqBottom = 0;

%% --- Top wave control ---
perturbTop = false;
ampTop = 0.01;
freqTop = 0;

%% Create bottom wave
xWaveBottom = linspace(xLeft, xRight, NwavePts);
yWaveBottom = yBottom + ampBottom * sin(freqBottom * 2*pi * (xWaveBottom - xLeft)/width);

%% Create top wave (optional)
if perturbTop
    % Reverse x-direction to maintain correct polygon orientation
    xWaveTop = linspace(xRight, xLeft, NwavePts);
    yWaveTop = yTop + ampTop * sin(freqTop * 2*pi * (xWaveTop - xLeft)/width);
else
    % Straight top edge if perturbation is disabled
    xWaveTop = [xRight, xLeft];
    yWaveTop = [yTop, yTop];
end

%% Combine coordinates for the inner hole
xHole = [xWaveBottom, xWaveTop];
yHole = [yWaveBottom, yWaveTop];

% Convert to MATLAB geometry matrix format
innerPoly = [2, numel(xHole), xHole, yHole]';

%% Combine outer square and inner hole into final geometry
maxRows = max(size(outerSquare,1), size(innerPoly,1));
outerSquare(end+1:maxRows,1) = 0;
innerPoly(end+1:maxRows,1)   = 0;

gd = [outerSquare, innerPoly];
ns = char('SQ1','RC1')';
sf = 'SQ1 - RC1';

[g, bt] = decsg(gd, sf, ns);
geometryFromEdges(model, g);


%% Apply boundary condition and define the PDE for the SCATTERED field u_scat

% --- Outer Boundary: Absorbing Boundary Condition (Robin) ---
applyBoundaryCondition(model, "neumann", ...
                       Edge=(1:4), ...
                       q = -1i*k, ...
                       g = 0);

% --- Inner Boundary (Obstacle): Total field is zero ---
%u_total = u_inc + u_scat = 0  =>  u_scat = -u_inc
% Plane wave
%incident_field = @(location,state) -exp(1i*k*location.y);

% Cylindrical wave
source = [0.5, -2.0];

% Define cylindrical incident field (and also add the minus sign)
incident_field = @(location,state) -50 * (1i/4) * besselh(0,1, ...
    k * sqrt((location.x - source(1)).^2 + (location.y - source(2)).^2) );

applyBoundaryCondition(model, "dirichlet", ...
                       Edge=(5:106), ...
                       u = incident_field);

% --- Define PDE coefficients ---
c = 1;
a = -k^2;
f = 0;
specifyCoefficients(model, m=0, d=0, c=c, a=a, f=f);

%% Mesh the geometry
generateMesh(model, Hmax=mesh_size);
figure
pdemesh(model);
axis equal
title('Problem Geometry and Mesh');

%% Solve for the scattered field u_scat
result = solvepde(model);
u_scat = result.NodalSolution;

%% Plot the results
nodes = model.Mesh.Nodes;
x = nodes(1, :);
y = nodes(2, :);

% Incident field at all node locations
u_inc_nodes = exp(1i * k * y);

% Total field is the sum of incident and scattered fields
u_total = u_scat + u_inc_nodes.';

figure;
pdeplot(model, 'XYData', abs(u_total), 'Mesh', 'off');
colormap("hot"); % Use 'hot' to match Python plot
colorbar;
xlabel('x');
ylabel('y');
title('Magnitude of Total Field |u_{inc} + u_{scat}|');
axis equal;

%% Take measurements
% Define edges and midpoints
% Define edges points where measurements are taken
x_edges = linspace(0, 1, 100);   % 100 points along x-axis
dx = x_edges(2) - x_edges(1);
x_meas_mid = x_edges(1:end-1) + dx/2;
y_meas_mid = zeros(size(x_meas_mid));  % y = 0 at the bottom edge

% Interpolate scattered field at edges
u_scat_mid = interpolateSolution(result, x_meas_mid, y_meas_mid);

% Calculate incident field at edges
u_inc_mid = exp(1i * k * y_meas_mid);

% Calculate u_total
%u_total_mid = u_scat_mid + u_inc_mid;
%u_total_mid_re = real(u_total_mid(:,1));
%u_total_mid_im = imag(u_total_mid(:,1));
%u_mag_mid = abs(u_total_mid(:,1));

u_scat_mid_re = real(u_scat_mid(:,1));
u_scat_mid_im = imag(u_scat_mid(:,1));
u_mag_mid = abs(u_scat_mid(:,1));

% Save (synthetic) measurements in a table
T = table(x_meas_mid.', y_meas_mid.', u_mag_mid, ...
          'VariableNames', {'x', 'y', 'u'});
% Write to CSV
writetable(T, 'matlab_measurements_sin0.5_scatter.csv');

% Save all data in another table for amplitude and phase error plot
T_complete = table(x_meas_mid.', y_meas_mid.', u_mag_mid, u_scat_mid_re, u_scat_mid_im, ...
          'VariableNames', {'x', 'y', 'mag_u', 'real_u', 'imag_u'});

writetable(T_complete, 'matlab_fullfield_sin0.5_scatter.csv');