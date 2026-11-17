function res = enclosure_kerr_alg1(opts)
%ENCLOSURE_KERR_ALG1  Algorithm 1 (enclosure method from simulated Neumann data).
%
%   Kow & Kuan, "An inverse problem for semilinear elliptic equations with
%   generalized Kerr-type nonlinearities", Section 4.
%
%   Model problem (4.1):  Delta u + chi_D |u|^2 u = 0 in B_1,  u = f on dB_1,
%   with D star-shaped about c, given in polar form
%       D = { c + s*r(phi)*(cos phi, sin phi) : 0 <= s < 1, 0 <= phi < 2 pi },
%   and compactly contained in B_1.  The default r(phi) = R gives a disk.
%
%   RES = ENCLOSURE_KERR_ALG1 runs Algorithm 1 with the default parameters
%   (h = 1/10, eps0 = 1/sqrt(2), lstar = 3, 36 directions) and plots the
%   result.  RES = ENCLOSURE_KERR_ALG1(OPTS) overrides any of the fields
%   h, eps0, lstar, Nw, c, R, rfun, Nr, Nphi, M, doPlot.
%
%   Example (non-convex trefoil containing the ball of radius 0.3 about c):
%     opts = struct('c', [-0.1 0.15], 'rfun', @(ph) 0.4 + 0.1*cos(3*ph), 'Nphi', 120);
%     res  = enclosure_kerr_alg1(opts);
%
%   Larger inclusion with smaller h (as in the accompanying runs):
%     opts = struct('h', 0.05, 'c', [0.05 0.05], 'rfun', @(ph) 0.55 + 0.15*cos(3*ph), ...
%                   'lstar', 3, 'Nr', 40, 'Nphi', 240, 'M', 1024);
%     opts = struct('h', 0.01, 'c', [0.05 0.05], 'rfun', @(ph) 0.55 + 0.15*cos(3*ph), ...
%                   'lstar', 1, 'Nr', 60, 'Nphi', 480, 'M', 2048);   % lstar = 1: see nl_bound
%
%   Forward solver.  The Picard iterates (4.3b) are computed through the
%   Dirichlet Green function of the unit disk,
%       W(x) = int_D G(x,y) F(y) dy,
%       d_nu W(xhat) = int_D P(xhat,y) F(y) dy,   P = Poisson kernel,
%   with Gauss-Legendre (in s) x trapezoidal (in phi) quadrature on D and
%   the trapezoidal rule on dB_1.  The log-singularity of G on the diagonal
%   is replaced by its cell average.  The normal derivative of v_fhat is
%   computed analytically: d_nu v_fhat = -(1/h) (xhat . rho) v_fhat.
%
%   Line 10 of Algorithm 1 forms the indicator from the simulated Neumann
%   data Lambda exactly as written, i.e. as the difference Lambda - d_nu v_f.
%   For comparison (diagnostics only, never used for the reconstruction)
%   the function also evaluates
%     I_bnd = eps0^4 int_{dB_1} d_nu W conj(fhat) dS     (avoids the cancellation
%             between Lambda and d_nu v_f, but the pairing with the oscillatory
%             fhat still loses a factor ~exp(-(1+t_*)/h); unreliable for h ~ 0.01)
%     I_vol = -eps0^4 int_D g(eps0^2 W + v) conj(v) dx    (identity (4.10b);
%             uses D explicitly, i.e. an "inverse crime")
%   the relative accuracy tau(omega) required of the Neumann data, and the
%   rigorous bound 3 eps0^2 |D| m^6 / K^(1) of Theorem 4.2(b),(c) on the
%   relative size of the nonlinear correction (m = exp(-(1+t_*)/h)).
%   The Green matrix (N x N) is built only when lstar >= 2.
%
%   Requires MATLAB R2016b or later (implicit expansion).

if nargin < 1, opts = struct(); end
p = setDefaults(opts);

h = p.h; eps0 = p.eps0; lstar = p.lstar; Nw = p.Nw; c = p.c(:).'; R = p.R;
if isempty(p.rfun)
    rfun = @(ph) R + 0*ph;                       % disk of radius R
else
    rfun = p.rfun;
end
assert(eps0^2 <= 1/2, 'Lemma 4.1 requires eps0^2 <= 1/2.');
phib = 2*pi*(0:4095)'/4096;                      % fine sampling of dD
bD   = c + rfun(phib).*[cos(phib), sin(phib)];
assert(max(sqrt(sum(bD.^2, 2))) < 1, 'D must be compactly contained in B_1.');

%% Quadrature on D:  y = c + s r(phi) e(phi),  dy = s r(phi)^2 ds dphi
[xg, wg] = gaussLegendre(p.Nr);
ss  = (xg + 1)/2;
ws  = wg/2;
phi = 2*pi*(0:p.Nphi-1)'/p.Nphi;
[Sg, PPg] = ndgrid(ss, phi);
RRg = Sg.*rfun(PPg);
Y   = [c(1) + RRg(:).*cos(PPg(:)), c(2) + RRg(:).*sin(PPg(:))];
wY  = reshape((ws.*ss)*ones(1, p.Nphi).*rfun(PPg).^2*(2*pi/p.Nphi), [], 1);
N   = size(Y, 1);
y2  = sum(Y.^2, 2);

%% Green matrix on D (only needed for Picard steps l >= 2):  GW(i,j) = G(Y_i, Y_j) * wY_j
GW = [];
if lstar >= 2
%  G(x,y) = (log|x-y| - log(|y| |x - y/|y|^2|)) / (2 pi),  Delta_x G = delta_y,
%  and |y| |x - y/|y|^2| = sqrt(|x|^2 |y|^2 - 2 x.y + 1).
D1 = sqrt((Y(:,1) - Y(:,1).').^2 + (Y(:,2) - Y(:,2).').^2);
D2 = sqrt(max(y2*y2.' - 2*(Y*Y.') + 1, 0));
D1(1:N+1:end) = 1;                              % placeholder, fixed below
GW = (log(D1) - log(D2))/(2*pi);
clear D1
a  = sqrt(wY/pi);                               % radius of a disk with the cell area
GW(1:N+1:end) = (log(a) - 0.5)/(2*pi) - log(diag(D2))/(2*pi);   % cell average
clear D2
GW = GW .* wY.';
end

%% Boundary quadrature and Poisson kernel:  PW(m,j) = P(X_m, Y_j) * wY_j
th = 2*pi*(0:p.M-1)'/p.M;
X  = [cos(th), sin(th)];
ds = 2*pi/p.M;
PW = (1 - y2.') ./ (2*pi*((X(:,1) - Y(:,1).').^2 + (X(:,2) - Y(:,2).').^2));
PW = PW .* wY.';

%% Algorithm 1
g     = @(z) abs(z).^2 .* z;
theta = 2*pi*(0:Nw-1)'/Nw;                       % line 2
[t_alg, t_bnd, t_vol, t_lin, t_star, rel_alg, tau, nl_bound] = deal(zeros(Nw, 1));
areaD = sum(wY);

for k = 1:Nw                                     % line 1
    om   = [cos(theta(k)), sin(theta(k))];
    omp  = [-sin(theta(k)), cos(theta(k))];
    rhoX = X*om.' + 1i*(X*omp.');                % rho . xhat on dB_1
    rhoY = Y*om.' + 1i*(Y*omp.');                % rho . y at the nodes in D

    fhat  = exp(-(1 + rhoX)/h);                  % line 3: f = eps0*fhat
    f     = eps0*fhat;
    vY    = exp(-(1 + rhoY)/h);                  % line 4: v_fhat at the nodes in D
    dvhat = -(1/h)*rhoX.*fhat;                   % d_nu v_fhat on dB_1 (nu = xhat)
    dvf   = eps0*dvhat;                          % d_nu v_f

    W = zeros(N, 1);                             % line 5: W^(0) = 0
    for l = 1:lstar                              % lines 6-8
        F = -g(eps0^2*W + vY);                   % right-hand side of (4.3b), uses W^(l-1)
        if l < lstar
            W = GW*F;                            % W^(l) at the nodes in D
        else
            dW    = PW*F;                        % d_nu W^(lstar) on dB_1
            Flast = F;
        end
    end

    Lambda = eps0*dvhat + eps0^3*dW;             % line 9: simulated measurement
    I_alg  = sum((Lambda - dvf).*conj(f))*ds;    % line 10: difference of Neumann data
    t_alg(k) = -1 - (h/4)*log(abs(I_alg)/eps0^4);   % line 11

    % ---- diagnostics (not part of Algorithm 1) ----
    I_bnd = eps0^4*sum(dW.*conj(fhat))*ds;
    I_vol = eps0^4*sum(Flast.*conj(vY).*wY);
    K1    = sum(abs(vY).^4.*wY);
    t_bnd(k)  = -1 - (h/4)*log(abs(I_bnd)/eps0^4);
    t_vol(k)  = -1 - (h/4)*log(abs(I_vol)/eps0^4);
    t_lin(k)  = -1 - (h/4)*log(K1);              % linearized level t^(1)_omega
    t_star(k) = min(bD*om.');                    % inf_{x in D} omega . x
    rel_alg(k) = abs(I_alg - I_vol)/abs(I_vol);
    tau(k) = abs(I_vol)/(sqrt(sum(abs(dvf).^2)*ds)*sqrt(sum(abs(f).^2)*ds));
    nl_bound(k) = exp(log(3*eps0^2*areaD) - 6*(1 + t_star(k))/h - log(K1));
end

%% Output
res = struct('theta', theta, 't_alg', t_alg, 't_bnd', t_bnd, 't_vol', t_vol, ...
             't_lin', t_lin, 't_star', t_star, 'rel_alg', rel_alg, 'tau', tau, ...
             'nl_bound', nl_bound, 'boundaryD', bD, 'areaD', areaD, 'params', p);

fprintf('\n  h = %g, eps0 = %.4f, lstar = %d, quadrature %d x %d on D, %d on dB_1\n', ...
        h, eps0, lstar, p.Nr, p.Nphi, p.M);
fprintf('\n  theta     t_*      t_alg     t_bnd     t_lin     tau        rel.err(line 10)  nonlin.bound\n');
for k = 1:Nw
    fprintf('  %5.3f  %+7.4f  %+8.4f  %+8.4f  %+8.4f  %9.2e  %9.2e         %9.2e\n', ...
        theta(k), t_star(k), t_alg(k), t_bnd(k), t_lin(k), tau(k), rel_alg(k), nl_bound(k));
end
fprintf('\n  unit roundoff = %.2e;  directions with tau < unit roundoff: %d of %d\n', ...
        eps/2, nnz(tau < eps/2), Nw);
fprintf(['  max |t_bnd - t_vol| = %.2e (agreement of boundary and volume forms;' ...
         ' degrades for small h)\n\n'], max(abs(t_bnd - t_vol)));

if p.doPlot
    plotResults(res);
end
end

%% ------------------------------------------------------------------------
function p = setDefaults(opts)
p = struct('h', 0.1, 'eps0', 1/sqrt(2), 'lstar', 3, 'Nw', 36, ...
           'c', [0.2, 0.1], 'R', 0.3, 'rfun', [], 'Nr', 40, 'Nphi', 96, ...
           'M', 1024, 'doPlot', true);
fn = fieldnames(opts);
for i = 1:numel(fn)
    p.(fn{i}) = opts.(fn{i});
end
end

function [x, w] = gaussLegendre(n)
% Gauss-Legendre nodes and weights on [-1,1] (Golub-Welsch).
k    = 1:n-1;
beta = k./sqrt(4*k.^2 - 1);
J    = diag(beta, 1) + diag(beta, -1);
[V, L] = eig(J);
[x, idx] = sort(diag(L));
w = 2*(V(1, idx).').^2;
end

function plotResults(res)
theta = res.theta;
bD    = res.boundaryD;
hullIdx = convhull(bD(:,1), bD(:,2));

% reconstructed convex hulls on a grid
n = 801;
[xx, yy] = meshgrid(linspace(-1, 1, n));
inB = xx.^2 + yy.^2 <= 1;
maskAlg = inB;  maskBnd = inB;  maskVol = inB;
for k = 1:numel(theta)
    s = cos(theta(k))*xx + sin(theta(k))*yy;
    maskAlg = maskAlg & (s >= res.t_alg(k));
    maskBnd = maskBnd & (s >= res.t_bnd(k));
    maskVol = maskVol & (s >= res.t_vol(k));
end

figure('Name', 'Algorithm 1: reconstruction');
hold on; axis equal; box on;
s = linspace(0, 2*pi, 400);
plot(cos(s), sin(s), 'k-');
fill(bD(:,1), bD(:,2), [0.85 0.85 0.85], 'EdgeColor', 'k');
plot(bD(hullIdx,1), bD(hullIdx,2), 'k:', 'LineWidth', 1.2);
contour(xx, yy, double(maskAlg), [0.5 0.5], 'r-', 'LineWidth', 1.5);
contour(xx, yy, double(maskBnd), [0.5 0.5], 'b--', 'LineWidth', 1.2);
contour(xx, yy, double(maskVol), [0.5 0.5], 'g:', 'LineWidth', 1.8);
legend({'\partial B_1', 'D', 'convex hull of D', 'D_{rec}, Algorithm 1 (line 10)', ...
        'D_{rec}, boundary-form diagnostic', ...
        'D_{rec}, exact-arithmetic reference (uses D)'}, 'Location', 'southoutside');
title(sprintf('h = %g, \\epsilon_0 = %.3f, \\ell^* = %d', ...
      res.params.h, res.params.eps0, res.params.lstar));
xlim([-1 1]); ylim([-1 1]);

figure('Name', 'Algorithm 1: levels and accuracy');
subplot(2, 1, 1);
plot(theta, res.t_star, 'k-', theta, res.t_lin, 'c-', theta, res.t_vol, 'g:', ...
     theta, res.t_bnd, 'b--', theta, res.t_alg, 'ro', 'LineWidth', 1);
legend({'t_*', 't^{(1)}_\omega', 't_\omega (reference, uses D)', ...
        't_\omega (boundary-form diagnostic)', 't_\omega (Algorithm 1)'}, ...
       'Location', 'best');
xlabel('\theta'); ylabel('level'); xlim([0 2*pi]); grid on;
subplot(2, 1, 2);
semilogy(theta, res.tau, 'k-', theta, res.rel_alg, 'ro', ...
         theta, (eps/2)*ones(size(theta)), 'b:', 'LineWidth', 1);
legend({'\tau(\omega)', 'relative error of line 10', 'unit roundoff'}, ...
       'Location', 'best');
xlabel('\theta'); xlim([0 2*pi]); grid on;
end
