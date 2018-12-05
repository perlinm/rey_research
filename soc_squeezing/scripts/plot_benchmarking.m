clear all;
fig_size = [ 6.5 5 ];

data_dir = "../data/model_benchmarking/";
fig_dir = "../figures/";
U_vals = importdata(data_dir + "U_range.dat");
phi_vals = importdata(data_dir + "phi_range.dat");
phi_vals = log10(phi_vals/pi);

[ U_vals, phi_vals ] = meshgrid(U_vals, phi_vals);
U_vals = transpose(U_vals);
phi_vals = transpose(phi_vals);


make_plot = @(N_tag, dtype) make_plot_full(data_dir, U_vals, phi_vals, N_tag, dtype);

figure;
set(gcf, "Visible", "off");

subplot(2,2,1); % top left
make_plot("12", "sq");

subplot(2,2,2); % top right
make_plot("09", "sq");

subplot(2,2,3); % bottom left
make_plot("12", "t");

subplot(2,2,4); % bottom right
make_plot("09", "t");

set(gcf, "Units", "inches");
set(gcf, "OuterPosition", [ 0 0 fig_size(1) fig_size(2) ]);
saveas(gcf, fig_dir + "model_benchmarking.png");

function make_plot_full(data_dir, U_vals, phi_vals, N_tag, dtype)
    color_FH = [ .1 .6 .85 ];
    color_OAT = [ .5 .5 .5 ];

    if dtype == "t"
        data_scale = 2*pi;
    else
        data_scale = 1;
    end
    FH = importdata(data_dir + "Hubbard12" + N_tag + "_" + dtype + ".dat") / data_scale;
    spin = importdata(data_dir + "Spin12" + N_tag + "_" + dtype + ".dat") / data_scale;
    OAT = importdata(data_dir + "OAT12" + N_tag + "_" + dtype + ".dat") / data_scale;
    OAT(1,:) = OAT(2,:);

    surf(U_vals, phi_vals, FH, "FaceAlpha", 1, "FaceColor", color_FH, "EdgeAlpha", 0); hold on
    surf(U_vals, phi_vals, spin, "FaceAlpha", 0, "EdgeAlpha", 1); hold on
    surf(U_vals, phi_vals, OAT, "FaceAlpha", 0.5, "FaceColor", color_OAT, "EdgeAlpha", 0);

    set(gca, "XLim", [0 8]);
    set(gca, "XTick", 0:2:8);
    set(gca, "YLim", [-2 -1]);
    set(gca, "YTick", -2:0.5:-1);
    xlabel("$U/J$", "interpreter", "latex");
    ylabel("$\log_{10}(\phi/\pi)$", "interpreter", "latex");
    if dtype == "sq"
        set(gca, "ZTick", 0:2:6);
        z_text = "Squeezing (dB)";
    else
        z_text = "Time ($2\pi/J$)";
    end
    zlabel(z_text, "interpreter", "latex");
end