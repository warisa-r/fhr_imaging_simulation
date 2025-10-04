%% Define constants
light_speed = 299792458;
frequency = 5e9;
k0 = 2 * pi * frequency / light_speed;  % Wave number in free space
wavelength = light_speed / frequency;
mesh_size = wavelength / 6;

%% Material properties
% Define different materials
k_air = k0;                    % Wave number in air (background)
n_obstacle = 1.5;              % Refractive index of obstacle material
k_obstacle = k0 * n_obstacle;  % Wave number in obstacle

%% Create a geometry model
model = createpde();

% Create outer square
outerSquare = [3 4 0 1 1 0 0 0 1 1]';

%% Wavy obstacle parameters
width = 0.4;
height = 0.2;
xCenter = 0.5;
yCenter = 0.5;
xLeft = xCenter - width/2;
xRight = xCenter + width/2;
yBottom = yCenter - height/2;
yTop = yCenter + height/2;
NwavePts = 100;

%% --- Bottom wave parameters ---
ampBottom = 0.01;
freqBottom = 0.5;

%% --- Top wave control ---
perturbTop = false;
ampTop = 0.01;
freqTop = 2;

%% Create bottom wave
xWaveBottom = linspace(xLeft, xRight, NwavePts);
yWaveBottom = yBottom + ampBottom * sin(freqBottom * 2*pi * (xWaveBottom - xLeft)/width);

%% Create top wave (optional)
if perturbTop
    xWaveTop = linspace(xRight, xLeft, NwavePts);
    yWaveTop = yTop + ampTop * sin(freqTop * 2*pi * (xWaveTop - xLeft)/width);
else
    xWaveTop = [xRight, xLeft];
    yWaveTop = [yTop, yTop];
end

%% Combine coordinates for the inner obstacle (now filled, not a hole)
xObstacle = [xWaveBottom, xWaveTop];
yObstacle = [yWaveBottom, yWaveTop];

% Convert to MATLAB geometry matrix format
innerPoly = [2, numel(xObstacle), xObstacle, yObstacle]';

%% Combine outer square and inner obstacle into final geometry
maxRows = max(size(outerSquare,1), size(innerPoly,1));
outerSquare(end+1:maxRows,1) = 0;
innerPoly(end+1:maxRows,1) = 0;
gd = [outerSquare, innerPoly];
ns = char('SQ1','OBS1')';  % Changed name from RC1 to OBS1
sf = 'SQ1 + OBS1';        % Changed from subtraction to addition to mesh both domain

[g, bt] = decsg(gd, sf, ns);
geometryFromEdges(model, g);

%% Apply boundary conditions
% Define rhs term function for absorbing boundary condition
function g = grad_u_inc_dot_n_minus_ik_u_inc(location, k)
    x = location.x;
    y = location.y;
    n_x = location.nx;
    n_y = location.ny;
    u_inc = exp(1i*k*y);
    du_inc_dx = 0;
    du_inc_dy = 1i*k*u_inc;
    dudn = du_inc_dx.*n_x + du_inc_dy.*n_y;
    g = dudn - 1i*k*u_inc;
end

% Outer Boundary: Absorbing Boundary Condition (Robin)
applyBoundaryCondition(model, 'neumann', ...
    'Edge', 1:4, ...  % Outer boundary edges
    'q', -1i*k_air, ...
    'g', @(location,state) grad_u_inc_dot_n_minus_ik_u_inc(location, k_air) );

%% Mesh the geometry first (needed to identify faces)
generateMesh(model, 'Hmax', mesh_size);

%% Plot geometry to identify face numbers
figure;
pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5);
title('Geometry with Face Labels');
axis equal;

%% Define PDE coefficients with domain-dependent wave numbers
% Check how many faces we have
num_faces = model.Geometry.NumFaces;
fprintf('Number of faces in geometry: %d\n', num_faces);

% Apply coefficients to each face
if num_faces == 1
    % Only one domain - uniform coefficients
    fprintf('Single domain detected, using uniform coefficients\n');
    specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', -k_air^2, 'f', 0);
else
    % Multiple domains - specify for each face
    fprintf('Multiple domains detected\n');
    
    % For air domain (face 1 - the background)
    specifyCoefficients(model, 'Face', 1, 'm', 0, 'd', 0, 'c', 1, 'a', -k_air^2, 'f', 0);
    fprintf('Applied air coefficients to face 1 (k = %.4f)\n', k_air);
    
    % For obstacle domain (face 2)
    if num_faces >= 2
        specifyCoefficients(model, 'Face', 2, 'm', 0, 'd', 0, 'c', 1, 'a', -k_obstacle^2, 'f', 0);
        fprintf('Applied obstacle coefficients to face 2 (k = %.4f)\n', k_obstacle);
    end
    
    % Handle additional faces if any
    for face_id = 3:num_faces
        % Default to air properties for additional faces
        specifyCoefficients(model, 'Face', face_id, 'm', 0, 'd', 0, 'c', 1, 'a', -k_air^2, 'f', 0);
        fprintf('Applied air coefficients to face %d\n', face_id);
    end
end

%% Display mesh information
figure;
subplot(1,2,1);
pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5);
title('Geometry with Face Labels');
axis equal;

subplot(1,2,2);
pdemesh(model);
axis equal;
title('Mesh');

%% Solve for the total field
result = solvepde(model);
u_total = result.NodalSolution;

%% Plot the results
nodes = model.Mesh.Nodes;
x = nodes(1, :);
y = nodes(2, :);

figure;
pdeplot(model, 'XYData', abs(u_total), 'Mesh', 'off');
colormap("hot"); % Use 'hot' to match Python plot
colorbar;
xlabel('x');
ylabel('y');
title('Magnitude of Total Field |u_{inc} + u_{scat}|');
axis equal;

%% Take measurements along y = 0
x_edges = linspace(0, 1, 100);
dx = x_edges(2) - x_edges(1);
x_meas_mid = x_edges(1:end-1) + dx/2;
y_meas_mid = zeros(size(x_meas_mid));

% Interpolate total field at measurement points
u_total_mid = interpolateSolution(result, x_meas_mid, y_meas_mid);

% Extract components
u_total_mid_re = real(u_total_mid(:,1));
u_total_mid_im = imag(u_total_mid(:,1));
u_mag_mid = abs(u_total_mid(:,1));

% Save measurements
T = table(x_meas_mid.', y_meas_mid.', u_mag_mid, u_total_mid_re, u_total_mid_im, ...
    'VariableNames', {'x', 'y', 'u', 'real_u', 'imag_u'});

% Write to CSV
%writetable(T, 'matlab_measurements_sin0.5_refraction_meshed_obstacle.csv');


%writetable(T_complete, 'matlab_fullfield_sin0.5_refraction.csv');