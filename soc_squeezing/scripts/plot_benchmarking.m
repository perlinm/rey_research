clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE CONTENTS:
% makes the model benchmarking figure for the SOC / squeezing paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set figure options
fig_size = [ 6.5 5.5 ];
color_FH = [ .1 .6 .8 ];
color_OAT = [ .5 .5 .5 ];
inset_cmap = [ 0 0 0 ; .2 .8 .2 ];
acceptable_error = 0.2;

% set directories
fig_dir = "../figures/";
data_dir = "../data/model_benchmarking/";

% import and process axis data
U_vec = importdata(data_dir + "U_range.dat");
phi_vec = importdata(data_dir + "phi_range.dat")/pi;
[U_vals,phi_vals] = meshgrid(U_vec, phi_vec);
U_vals = transpose(U_vals);
phi_vals = transpose(phi_vals);

% shorthand for the functions to make subplots and insets
make_subplot = @(N_tag, dtype, title_text) ...
    make_subplot_full(N_tag, dtype, title_text, ...
                      data_dir, U_vals, phi_vals, color_FH, color_OAT, acceptable_error);
make_inset = @(position, acceptable_params) ...
    make_inset_full(position, acceptable_params, U_vec, phi_vec, inset_cmap);

% prepare empty summary of acceptable parameter regimes
[xs,ys] = size(U_vals);
acceptable_params = zeros(xs, ys, 4);

% make all surface plots
set(gcf, 'Visible', 'off');
subplot_args = { { '12' 'sq' '{\bf (a.i)} $f=1$' } ...
                 { '09' 'sq' '{\bf (b.i)} $f=3/4$' } ...
                 { '12' 't' '{\bf (a.ii)} $f=1$' } ...
                 { '09' 't' '{\bf (b.ii)} $f=3/4$' } };
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
    legendflex(bars, {'FH', 'Spin', 'OAT'}, 'interpreter', 'latex', ...
               'ref', gcf, 'anchor', {'ne' 'ne'});
hatchfill2(patch_obj(length(bars)+2), 'cross', 'HatchAngle', 0, 'HatchDensity', 6);

% make insets showing "acceptable" parameter regimes
width = 0.1;
height = 0.12;
bottom = 0.5 - height/2;
shift = 0.3;
ax_left = subplot(2,2,3);
ax_right = subplot(2,2,4);
offset_lt = ax_left.Position(1);
offset_rt = ax_right.Position(1);
make_inset([ offset_lt+shift  bottom width height], ...
           acceptable_params(:,:,1) & acceptable_params(:,:,3));
make_inset([ offset_rt+shift bottom width height], ...
           acceptable_params(:,:,2) & acceptable_params(:,:,4));

% scale figure properly and save it
set(gcf, 'Units', 'inches');
set(gcf, 'OuterPosition', [ 0 0 fig_size(1) fig_size(2) ]);
set(gcf, 'Renderer', 'Painters');
fig_pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [fig_pos(3) fig_pos(4)]);
print(fig_dir + 'model_benchmarking', '-dpdf');
clf;

% function to make one surface plot comparing FH, spin, and OAT models
function [ acceptable_params ] = ...
    make_subplot_full(N_text, dtype, title_text, data_dir, U_vals, phi_vals, ...
                      color_FH, color_OAT, acceptable_error)
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

    % collect output data
    data_max = max(max(data_FH, data_spin), data_OAT);
    data_min = min(min(data_FH, data_spin), data_OAT);
    acceptable_params = 1 - data_min./data_max < acceptable_error;

    % make all surface plots
    surf(U_vals, phi_vals, data_FH, 'FaceColor', color_FH, 'EdgeColor', 'none'); hold on;
    mesh(U_vals, phi_vals, data_spin, 'FaceAlpha', 0, 'EdgeColor', [0 0 0]);
    surf(U_vals, phi_vals, data_OAT, 'FaceColor', color_OAT, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    set(gca, 'YScale', 'Log');

    % set axis ticks and labels
    set(gca, 'XLim', [0 8], 'XTick', 0:2:8);
    set(gca, 'YLim', [1/100 1/10], 'YTick', [1/100 1/30 1/10]);
    set(gca, 'YTickLabel', {'$\pi/100$', '$\pi/30$', '$\pi/10$'});
    set(gca, 'TickLabelInterpreter', 'latex');
    if strcmp(dtype, 't')
        xlabel('$U/J$', 'interpreter', 'latex');
        ylabel_obj = ylabel('$\phi$', 'interpreter', 'latex');
        ylabel_obj.Units = 'normalized';
        ylabel_obj.Position = ylabel_obj.Position + [0 0.02 0];
        z_max = max(max(data_max));
        set(gca, 'ZLim', [0 z_max], 'ZTick', 0:5:z_max);
        text(0.1, 1/10, z_max*1.05, '$\times10^2$', 'interpreter', 'latex');
    else % if dtype == 'sq'
        set(gca, 'ZLim', [0 6], 'ZTick', 0:2:6);
    end
    if strcmp(N_text, '12')
        if strcmp(dtype, 'sq')
            z_text = 'Squeezing (dB)';
            dz = [ -0.03 0.07 0 ];
        else % dtype == 't'
            z_text = 'Time ($2\pi/J$)';
            dz = [ 0 0.03 0 ];
        end
        zlabel_obj = zlabel(z_text, 'interpreter', 'latex');
        zlabel_obj.Units = 'normalized';
        zlabel_obj.Position = zlabel_obj.Position + dz;
    end

    % set plot title
    title_obj = title(title_text, 'interpreter', 'latex');
    title_obj.Units = 'normalized';
    title_obj.Position = [ .2 .95 ];
end

function make_inset_full(position, acceptable_params, U_vec, phi_vec, cmap)
    ax = axes('Position', position);
    set(ax, 'FontSize', 8);
    imagesc(U_vec, phi_vec, transpose(acceptable_params));
    set(gca, 'YDir', 'normal');
    colormap(ax, cmap);

    set(ax, 'YScale', 'Log');
    set(ax, 'XLim', [0 8], 'XTick', 0:2:8, 'XTickLabel', {'0','','','','8'});
    set(ax, 'YLim', [1/100 1/10], 'YTick', [1/100 1/10], 'YTickLabel', {'$\pi/100$', '$\pi/10$'});
    set(ax, 'TickLabelInterpreter', 'latex');
    set(ax, 'TickDir', 'out', 'TickLength', [0.03 0]);

    xlabel_obj = xlabel('$U/J$', 'interpreter', 'latex');
    ylabel_obj = ylabel('$\phi$', 'interpreter', 'latex');
    set(ylabel_obj, 'rotation', 0, 'VerticalAlignment','middle');
    xlabel_obj.Units = 'normalized';
    ylabel_obj.Units = 'normalized';
    xlabel_obj.Position = xlabel_obj.Position + [ 0 0.1 0 ];
    ylabel_obj.Position = ylabel_obj.Position + [ 0.25 0 0 ];
end
