clear all;

% set figure options
fig_size = [ 6.5 5 ];
color_FH = [ .1 .6 .85 ];
color_OAT = [ .5 .5 .5 ];
inset_cmap = [0 0 0;1 1 0];
data_error = 0.15;

% set directories
fig_dir = "../figures/";
data_dir = "../data/model_benchmarking/";

% import and process axis data
U_vals = importdata(data_dir + "U_range.dat");
phi_vals = importdata(data_dir + "phi_range.dat")/pi;
[U_vals,phi_vals] = meshgrid(U_vals, phi_vals);
U_vals = transpose(U_vals);
phi_vals = transpose(phi_vals);

% shorthand for the functions to make subplots and insets
make_subplot = @(N_tag, dtype, title_text) ...
    make_subplot_full(N_tag, dtype, title_text, ...
                      data_dir, U_vals, phi_vals, color_FH, color_OAT, data_error);
make_inset = @(position, acceptable_params) ...
    make_inset_full(position, acceptable_params, U_vals, phi_vals, inset_cmap);

% prepare empty summary of acceptable parameter regimes
[xs,ys] = size(U_vals);
acceptable_params = zeros(xs, ys, 4);

% make all surface plots
figure;
set(gcf, 'Visible', 'off');
subplot_args = { { '12' 'sq' '(a.i)' } ...
                 { '09' 'sq' '(b.i)' } ...
                 { '12' 't' '(a.ii)' } ...
                 { '09' 't' '(b.ii)' } };
for nn = 1:length(subplot_args)
    subplot(2,2,nn);
    acceptable_params(:,:,nn) = make_subplot(subplot_args{nn}{:});
end

% make legend for primary subplots
bars = bar(rand(2,3),'Visible','off');
set(bars(1), 'FaceColor', color_FH, 'EdgeAlpha', 0);
set(bars(2), 'FaceColor', 'w');
set(bars(3), 'FaceColor', color_OAT, 'EdgeAlpha', 0, 'FaceAlpha', 0.5);
[legend_obj,patch_obj,~,~] = ...
    legendflex(bars, {'FH', 'Spin', 'OAT'}, ...
                      'ref', gcf, 'anchor', {'s' 's'}, ...
                      'bufferunit', 'normalized', 'buffer', [0 0.45]);
hatchfill2(patch_obj(length(bars)+2), 'cross', 'HatchAngle', 0, 'HatchDensity', 6);

% make insets showing "acceptable" parameter regimes
width = 0.1;
height = 0.1;
pad = 0.025;
bottom = 0.5 - height/2;
make_inset([pad bottom width height], ...
           acceptable_params(:,:,1) & acceptable_params(:,:,3));
make_inset([1-width-pad bottom width height], ...
           acceptable_params(:,:,2) & acceptable_params(:,:,4));

% scale figure properly and save it to a file
set(gcf, 'Units', 'inches');
set(gcf, 'OuterPosition', [ 0 0 fig_size(1) fig_size(2) ]);
print(fig_dir + 'model_benchmarking.png', '-dpng', '-r600');

% function to make one surface plot comparing FH, spin, and OAT models
function [ acceptable_params ] = ...
    make_subplot_full(N_text, dtype, title_text, data_dir, U_vals, phi_vals, ...
                      color_FH, color_OAT, data_error)
    % scale data by 2\pi if we are plotting squeezing times
    if strcmp(dtype, 't')
        data_scale = 2*pi * 100;
    else
        data_scale = 1;
    end
    
    % import all data
    data_FH = importdata(data_dir + 'Hubbard12' + N_text + '_' + dtype + '.dat') / data_scale;
    data_spin = importdata(data_dir + 'Spin12' + N_text + '_' + dtype + '.dat') / data_scale;
    data_OAT = importdata(data_dir + 'OAT12' + N_text + '_' + dtype + '.dat') / data_scale;
    data_OAT(1,:) = data_OAT(2,:); % correct for an artifact of convention at U/J = 0

    % make all surface plots
    surf(U_vals, phi_vals, data_FH, 'FaceColor', color_FH, 'EdgeAlpha', 0); hold on;
    mesh(U_vals, phi_vals, data_spin, 'FaceAlpha', 0, 'EdgeColor', [0 0 0]);
    surf(U_vals, phi_vals, data_OAT, 'FaceColor', color_OAT, 'EdgeAlpha', 0, 'FaceAlpha', 0.5);
    set(gca, 'YScale', 'Log');

    % set axis ticks and labels
    set(gca, 'XLim', [0 8], 'XTick', 0:2:8);
    set(gca, 'YLim', [1/100 1/10], 'YTick', [1/100 1/30 1/10]);
    set(gca, 'YTickLabel', {'$\pi/100$', '$\pi/30$', '$\pi/10$'});
    set(gca, 'TickLabelInterpreter', 'latex');
    xlabel('$U/J$', 'interpreter', 'latex');
    ylabel_obj = ylabel('$\phi$', 'interpreter', 'latex');
    ylabel_obj.Units = 'normalized';
    ylabel_obj.Position = ylabel_obj.Position + [0 0.02 0];
    if strcmp(dtype, 'sq')
        set(gca, 'ZTick', 0:2:6);
        z_text = 'Squeezing (dB)';
    else
        z_text = 'Time ($2\pi/J$)';
        zlim = get(gca, 'ZLim');
        text(-1, 1/10, zlim(2)*1.2, '$\times10^2$', 'interpreter', 'latex');
    end
    zlabel_obj = zlabel(z_text, 'interpreter', 'latex');
    
    % adjust position of the vertical axis labels
    if strcmp(dtype, 'sq')
        dz = [ -0.03 0.07 0 ];
    else
        dz = [ 0 0.03 0 ];
    end
    zlabel_obj.Units = 'normalized';
    zlabel_obj.Position = zlabel_obj.Position + dz;
    
    % set plot title
    title_obj = title(title_text);
    title_obj.Units = 'normalized';
    title_obj.Position = [ .2 .95 ];

    % collect output data
    data_max = max(max(data_FH, data_spin), data_OAT);
    data_min = min(min(data_FH, data_spin), data_OAT);
    acceptable_params = 1 - data_min./data_max < data_error;
end

function make_inset_full(position, acceptable_params, U_vals, phi_vals, cmap)
    ax = axes('Position', position);
    inset = pcolor(U_vals, phi_vals, double(acceptable_params));
    set(inset, 'EdgeAlpha', 0);
    set(ax, 'XTick', [], 'XTickLabel', {});
    set(ax, 'YTick', [], 'YTickLabel', {});
    colormap(ax, cmap);
end