clc;
clear;

%% Load data

gm = stlread('.\man_of_steel_statisch.stl');
load('.\radar_parameters');

%% Show 3D torso and antenna positions

P_gm = gm.Points*1e-3;
E_gm = gm.ConnectivityList;

figure;
scatter3(ant_pos(3,:), ant_pos(1,:), ant_pos(2,:));
hold on;
scatter3(ant_pos(6,:), ant_pos(4,:), ant_pos(5,:));
legend('TX', 'RX');
patch('Vertices', circshift(P_gm, 1, 2), 'Faces', E_gm, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.15);
hold off;
axis equal;


%% Reduce to 2D problem

y_TX = -0.08;
y_RX = 0.1;

y0 = 1/2*(y_TX + y_RX);

idx = ant_pos(2,:) == 0 & ant_pos(5,:) == 0.1;

ant_pos_2D = ant_pos([1, 3, 4,6], idx);

%% Calculate surface 

x = [-0.25:0.001:0.25];
y = y0;

[X_torso ,Y_torso] = ndgrid(x,y);

P = [X_torso(:), Y_torso(:)];

Z_torso = Inf(size(X_torso));

% Find object's z-coordinates corresponding to the x-y surface grid via
% calculating barycentric coordinates of the projected object grid (in x-y-
% plane)

for jj = 1:size(E_gm,1)
    P_a = P_gm(E_gm(jj,1),:);
    P_b = P_gm(E_gm(jj,2),:);
    P_c = P_gm(E_gm(jj,3),:);
    
    beta = ( ( P_a(1)  - P_c(1) ) .* (P(:,2) - P_c(2)) - (P_a(2)  - P_c(2)) .* (P(:, 1) - P_c(1)) ) ./ ...
        ( (P_b(1)  - P_c(1)) .* (P_c(2) - P_a(:,2)) - (P_b(2)  - P_c(2)) .* (P_c(1) - P_a(1)) );
    
    gamma = ( (P_b(1)  - P_a(1)) .* (P(:, 2) - P_a(2)) - (P_b(2)  - P_a(2)) .*  (P(:, 1) - P_a(1)) ) ./ ...
        ( (P_b(1)  - P_a(1)) .* (P_c(2) - P_a(2)) - (P_b(2)  - P_a(2)) .*   (P_c(1) - P_a(1)) );  
        
    alpha = 1 - beta - gamma;


    % plot(alpha .* P_a(1) + beta .* P_b(1) + gamma .* P_c(1) - P(:, 1))
    % plot(alpha .* P_a(2) + beta .* P_b(2) + gamma .* P_c(2) - P(:, 2))

    idx_jj = alpha >= 0& beta >= 0 & gamma >= 0;

    Z_torso(idx_jj) = min(Z_torso(idx_jj), alpha(idx_jj) .* P_a(3) + beta(idx_jj) .* P_b(3) + gamma(idx_jj) .* P_c(3));
end

%% Show 3D setting

[X_surf, Z_surf] = ndgrid(x, 0:0.01:0.8);
Y_surf = y0*ones(size(X_surf));


figure;
scatter3(ant_pos(3,:), ant_pos(1,:), ant_pos(2,:));
hold on;
scatter3(ant_pos(6,:), ant_pos(4,:), ant_pos(5,:));
legend('TX', 'RX');
patch('Vertices', circshift(P_gm, 1, 2), 'Faces', E_gm, 'FaceAlpha', 0.2, 'EdgeAlpha', 0.15);
mesh(Z_surf, X_surf, Y_surf, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.1);
plot3(Z_torso, X_torso,  Y_torso, 'LineWidth', 2);
hold off;
axis equal;


%% Show reduced 2D setting

torso_2D_bd = [X_torso(:).'; Z_torso(:).'];

figure;
scatter(ant_pos_2D(1,:), ant_pos_2D(2,:), '*');
hold on;
scatter(ant_pos_2D(3,:), ant_pos_2D(4,:));
plot(torso_2D_bd(1,:), torso_2D_bd(2,:));
hold off;
axis equal;

%%

