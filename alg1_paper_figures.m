%ALG1_PAPER_FIGURES  Figures for Section 4 (Algorithm 1, h = 0.05, trefoil inclusion).
%
%   Produces
%     fig_alg1_reconstruction.pdf : dB_1, D, convex hull of D, D_rec from Algorithm 1,
%                                   and the reference D_rec from identity (4.10b)
%     fig_alg1_levels.pdf         : t_*, t^(1)_omega, t^(lstar)_omega (Algorithm 1)
%                                   and the reference t^(lstar)_omega from (4.10b)
%   The reference evaluates the indicator through the volume identity (4.10b),
%   which uses D explicitly; it shows what line 10 would give in exact arithmetic.
%   Requires enclosure_kerr_alg1.m on the path (MATLAB R2016b or later).

clear; close all;

opts = struct('h', 0.05, 'eps0', 1/sqrt(2), 'lstar', 3, 'Nw', 36, ...
              'c', [0.05 0.05], 'rfun', @(ph) 0.55 + 0.15*cos(3*ph), ...
              'Nr', 40, 'Nphi', 240, 'M', 1024, 'doPlot', false);
res = enclosure_kerr_alg1(opts);

theta = res.theta;
bD    = res.boundaryD;
k     = convhull(bD(:,1), bD(:,2));

%% Reconstruction
n = 1201;
[xx, yy] = meshgrid(linspace(-1, 1, n));
mask = xx.^2 + yy.^2 <= 1;
maskRef = mask;
for j = 1:numel(theta)
    sj = cos(theta(j))*xx + sin(theta(j))*yy;
    mask    = mask    & (sj >= res.t_alg(j));
    maskRef = maskRef & (sj >= res.t_vol(j));
end

s = linspace(0, 2*pi, 600);
f1 = figure('Units', 'inches', 'Position', [1 1 4.6 5.4]);
hold on; box on; axis equal;
h1 = plot(cos(s), sin(s), 'k-', 'LineWidth', 0.9);
h2 = fill(bD(:,1), bD(:,2), [0.85 0.85 0.85], 'EdgeColor', 'k', 'LineWidth', 0.9);
h3 = plot(bD(k,1), bD(k,2), 'k:', 'LineWidth', 1.2);
[~, h5] = contour(xx, yy, double(maskRef), [0.5 0.5], 'g--', 'LineWidth', 1.3);
[~, h4] = contour(xx, yy, double(mask), [0.5 0.5], 'r-', 'LineWidth', 1.5);
xlim([-1.02 1.02]); ylim([-1.02 1.02]);
legend([h1 h2 h3 h4 h5], {'$\partial B_1$', '$D$', 'convex hull of $D$', ...
       '$D_{\mathrm{rec}}$ (Algorithm 1)', '$D_{\mathrm{rec}}$ via (4.10b) (reference)'}, ...
       'Interpreter', 'latex', ...
       'Location', 'southoutside', 'NumColumns', 2, 'FontSize', 8, 'Box', 'off');
exportgraphics(f1, 'fig_alg1_reconstruction.pdf', 'ContentType', 'vector');

%% Levels
f2 = figure('Units', 'inches', 'Position', [1 1 5.6 3.9]);
hold on; box on; grid on;
plot(theta, res.t_star, 'k-', 'LineWidth', 1.2);
plot(theta, res.t_lin, 'k--', 'LineWidth', 0.9);
plot(theta, res.t_vol, 'g:', 'LineWidth', 2.0);
plot(theta, res.t_alg, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
xlim([0 2*pi]);
xticks(pi*(0:0.5:2));
xticklabels({'$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('$\theta$ \ ($\omega = (\cos\theta, \sin\theta)$)', 'Interpreter', 'latex');
ylabel('level');
legend({'$t_*$', '$t^{(1)}_\omega$', '$t^{(\ell^*)}_\omega$ via (4.10b) (reference)', ...
        '$t^{(\ell^*)}_\omega$ (Algorithm 1)'}, ...
       'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 2, 'FontSize', 8, 'Box', 'off');
exportgraphics(f2, 'fig_alg1_levels.pdf', 'ContentType', 'vector');

%% Numbers quoted in the text
err = abs(res.t_alg - res.t_vol);
fprintf('bias t^(1) - t_*   : %.3f to %.3f\n', min(res.t_lin - res.t_star), max(res.t_lin - res.t_star));
fprintf('|t^(l*) - exact|   : max %.3f; < 1e-3 in %d of %d directions\n', ...
        max(err), nnz(err < 1e-3), numel(theta));
fprintf('t^(l*) - t_*       : %.3f to %.3f\n', min(res.t_alg - res.t_star), max(res.t_alg - res.t_star));
