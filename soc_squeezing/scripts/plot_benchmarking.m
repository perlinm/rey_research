clear all;

% set figure size and directories
fig_size = [ 6.5 5 ];
fig_dir = "../figures/";
data_dir = "../data/model_benchmarking/";

% import and process axis data
U_vals = importdata(data_dir + "U_range.dat");
phi_vals = importdata(data_dir + "phi_range.dat");
phi_vals = log10(phi_vals/pi);
[ U_vals, phi_vals ] = meshgrid(U_vals, phi_vals);
U_vals = transpose(U_vals);
phi_vals = transpose(phi_vals);

% shorthand for the function which makes surface plots
make_plot = @(N_tag, dtype, title_text) ...
    make_plot_full(N_tag, dtype, title_text, data_dir, U_vals, phi_vals);

% make all surface plots
figure;
set(gcf, 'Visible', 'off');
subplot(2,2,1); % top left
make_plot('12', 'sq', '(a.i)');
subplot(2,2,2); % top right
make_plot('09', 'sq', '(b.i)');
subplot(2,2,3); % bottom left
make_plot('12', 't', '(a.ii)');
subplot(2,2,4); % bottom right
make_plot('09', 't', '(b.ii)');

% make legend
subplot(2,2,1); % top left
leg = legend('FH', 'Spin', 'OAT');
leg.Location = 'east';
leg.Units = 'normalized';
width = leg.Position(3);
height = leg.Position(4);
left = 0.5 - width/2;
bottom = 0.5 - height/2;
leg.Position = [ left bottom width height ];

% scale figure properly and save it to a file
set(gcf, 'Units', 'inches');
set(gcf, 'OuterPosition', [ 0 0 fig_size(1) fig_size(2) ]);
print(fig_dir + 'model_benchmarking.png', '-dpng', '-r600');

% function to make one surface plot comparing FH, spin, and OAT models
function make_plot_full(N_tag, dtype, title_text, data_dir, U_vals, phi_vals)
    % set surface colors for each model
    color_FH = [ .1 .6 .85 ];
    color_spin = [ 1 1 1 ];
    color_OAT = [ .5 .5 .5 ];

    % scale data by 2\pi if we are plotting squeezing times
    if dtype == 't'
        data_scale = 2*pi;
    else
        data_scale = 1;
    end
    
    % import all data
    data_FH = importdata(data_dir + 'Hubbard12' + N_tag + '_' + dtype + '.dat') / data_scale;
    data_spin = importdata(data_dir + 'Spin12' + N_tag + '_' + dtype + '.dat') / data_scale;
    data_OAT = importdata(data_dir + 'OAT12' + N_tag + '_' + dtype + '.dat') / data_scale;
    data_OAT(1,:) = data_OAT(2,:); % correct for an artifact of convention at U/J = 0

    % make all surface plots
    surf(U_vals, phi_vals, data_FH, 'FaceColor', color_FH, 'FaceAlpha', 1, 'EdgeAlpha', 0); hold on;
    surf(U_vals, phi_vals, data_spin, 'FaceColor', color_spin, 'FaceAlpha', 0, 'EdgeAlpha', 1);
    surf(U_vals, phi_vals, data_OAT, 'FaceColor', color_OAT, 'FaceAlpha', 0.5, 'EdgeAlpha', 0);

    % set axis ticks and labels
    set(gca, 'XTick', 0:2:8);
    set(gca, 'YLim', [-2 -1]);
    set(gca, 'YTick', -2:0.5:-1);
    xlabel('$U/J$', 'interpreter', 'latex');
    ylabel_obj = ylabel('$\log_{10}(\phi/\pi)$', 'interpreter', 'latex');
    if dtype == 'sq'
        set(gca, 'ZTick', 0:2:6);
        z_text = 'Squeezing (dB)';
    else
        z_text = 'Time ($2\pi/J$)';
    end
    zlabel_obj = zlabel(z_text, 'interpreter', 'latex');
    
    % adjust position of axis labels
    ylabel_obj.Units = 'normalized';
    ylabel_obj.Position = [ ylabel_obj.Position(1)-0.05 ylabel_obj.Position(2)+0.1 ];
    if dtype == 'sq'
        zlabel_obj.Units = 'normalized';
        zlabel_obj.Position = [ zlabel_obj.Position(1)-0.04 zlabel_obj.Position(2) ];
    end
    
    % set plot title
    title_obj = title(title_text);
    title_obj.Units = 'normalized';
    title_obj.Position = [ .18 .9 ];
end