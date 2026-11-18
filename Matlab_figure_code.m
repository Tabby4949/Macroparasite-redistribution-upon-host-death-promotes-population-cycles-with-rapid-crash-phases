clear; clc;

%% Main paper: Figure 4

par.b      = 0.6;
par.d      = 0.5;
par.beta   = 0.12;
par.psi    = 0.5;
par.sigma  = 4;
par.xi     = 1e-4; 
par.delta  = 60;
par.nu     = 1e-4; 
par.h      = 100;
par.alpha  = 5e-4;
par.lambda = 40;

tspan = [0 4000];
opts = odeset('RelTol', 1e-5,'AbsTol', 1e-8,'NonNegative', [1, 2, 3, 4],'NormControl', 'on','MaxOrder', 2);

% Simulation: r = 0

par.r = 0;

y0 = initial_guess(par);
y0_r0 = y0 .* [1.1; 0.9; 1.1; 0.9];

[t0, y0] = ode23s(@(t, y) model(t, y, par), tspan, y0_r0, opts);

H0 = y0(:, 1);
x0 = y0(:, 2);
A0 = y0(:, 3);
L0 = y0(:, 4);

% Simulation: r = 1

par.r = 1;

y1 = initial_guess(par);
y1_r1 = y1 .* [1.1; 0.9; 1.1; 0.9];

[t1, y1] = ode23s(@(t, y) model(t, y, par), tspan, y1_r1, opts);

H1 = y1(:, 1);
x1 = y1(:, 2);
A1 = y1(:, 3);
L1 = y1(:, 4);

% Plot Trajectories

fig = figure;
fig.Color = [1 1 1];

tiledlayout(5, 1)

fSize = 15.5;

nexttile

hH1 = plot(t1, H1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hH0 = plot(t0, H0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([3700 3915]);
ylabel('H(t)', 'FontSize', 20);
grid off;

nexttile

hP1 = plot(t1, x1 .* H1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hP0 = plot(t0, x0 .* H0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([3700 3915]);
ylabel('P(t)', 'FontSize', 20);
grid off;

legend([hP0, hP1], {'r = 0', 'r = 1'}, 'Orientation', 'horizontal', 'Location', 'northeast', 'FontSize', 16);

nexttile

hx1 = plot(t1, x1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hx0 = plot(t0, x0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([3700 3915]);
ylabel('x(t)', 'FontSize', 20);
grid off;

nexttile

hL1 = plot(t1, L1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hL0 = plot(t0, L0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([3700 3915]);
ylabel('L(t)', 'FontSize', 20);
grid off;

nexttile

hA1 = plot(t1, A1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hA0 = plot(t0, A0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([3700 3915]);
xlabel('Time t', 'FontSize', 20);
ylabel('A(t)', 'FontSize', 20);
grid off;

%% Main paper: Figure 5

par.b      = 0.6;
par.d      = 0.5;
par.beta   = 0.12;
par.psi    = 0.5;
par.sigma  = 4;
par.xi     = 1e-4; 
par.delta  = 60;
par.nu     = 1e-4; 
par.h      = 100;
par.alpha  = 5e-4;
par.lambda = 40;

tspan = [0 30];
opts = odeset('RelTol', 1e-5,'AbsTol', 1e-8,'NonNegative', [1, 2, 3, 4],'NormControl', 'on','MaxOrder', 2);

% Simulation: r = 0

par.r = 0;

y0_r0 = [24.95; 0; 15.74; 0];

[t0, y0] = ode23s(@(t, y) model(t, y, par), tspan, y0_r0, opts);

H0 = y0(:, 1);
x0 = y0(:, 2);
A0 = y0(:, 3);
L0 = y0(:, 4);

% Simulation: r = 1
par.r = 1;

y1_r1 = [24.95; 0; 15.74; 0];

[t1, y1] = ode23s(@(t, y) model(t, y, par), tspan, y1_r1, opts);

H1 = y1(:, 1);
x1 = y1(:, 2);
A1 = y1(:, 3);
L1 = y1(:, 4);

% Plot Trajectories

fig = figure;
fig.Color = [1 1 1];

tiledlayout(5, 1)

fSize = 15.5;

nexttile

hH1 = plot(t1, H1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hH0 = plot(t0, H0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([19 23]);
ylabel('H(t)', 'FontSize', 20);
grid off;

nexttile

hP1 = plot(t1, x1 .* H1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hP0 = plot(t0, x0 .* H0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([19 23]);
ylabel('P(t)', 'FontSize', 20);
grid off;

legend([hP0, hP1], {'r = 0', 'r = 1'}, 'Orientation', 'horizontal', 'Location', 'northeast', 'FontSize', 16);

nexttile

hx1 = plot(t1, x1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hx0 = plot(t0, x0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([19 23]);
ylabel('x(t)', 'FontSize', 20);
grid off;

nexttile

hL1 = plot(t1, L1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hL0 = plot(t0, L0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim([19 23]);
ylabel('L(t)', 'FontSize', 20);
grid off;

nexttile

hA1 = plot(t1, A1, '-', 'LineWidth', 4, 'Color', [0 0 1]); hold on
hA0 = plot(t0, A0, ':', 'LineWidth', 4, 'Color', [1 0 0.25]);

set(gca, 'FontSize', fSize);
xlim(tspan);
xlim([19 23]);
xlabel('Time t', 'FontSize', 20);
ylabel('A(t)', 'FontSize', 20);
grid off;

%% Model

function dydt = model(~, y, par)
    H = y(1);
    x = y(2);
    A = y(3);
    L = y(4);
    
    H = max(H, 1e-12);
    x = max(x, 1e-12);

    dH = H * (par.b * (1+(A-1)*par.xi)^(-x/(A-1)) - par.d - par.nu * H - par.alpha * x);
    dx = par.beta * par.psi * L - x*(par.b*(1+(A-1)*par.xi)^(-x/(A-1))+par.sigma+par.alpha*A);
    dA = (1-A)*(par.sigma+par.alpha*A+(par.beta*par.psi*L/x))+par.b*x*(1+(A-1)*par.xi)^(-x/(A-1)) + (par.beta*par.psi*L*par.lambda)/x;
    dL = par.h * x * H - par.delta * L - par.beta * L * H + par.r * x * H * (par.d + par.nu * H + par.alpha * (A + x));
    
    dydt = [dH; dx; dA; dL];
end


function equilibrium = initial_guess(par)

    d      = par.d;
    beta   = par.beta;
    psi    = par.psi;
    sigma  = par.sigma;
    b      = par.b;
    nu     = par.nu;
    xi     = par.xi;
    alpha  = par.alpha;
    h      = par.h;
    delta  = par.delta;
    lambda = par.lambda;
    r      = par.r;

    if b <= d
        error('No endemic equilibrium possible because b <= d.');
    end

    % A(t) equilibrium equation

    A_eq = @(A,x) 2*alpha*A.^2 - A.*(alpha*(2 + lambda) + xi*x - 2*sigma - b) + x.*(xi*(1 + lambda + x) - b) - b*(1 + lambda) - sigma*(2 + lambda);

    % Find physical A equilibrium for a given x

    function A = A_root(x)

        a = 2*alpha;

        B = -(alpha*(2 + lambda) + xi*x - 2*sigma - b);

        c = x*(xi*(1 + lambda + x) - b) - b*(1 + lambda) - sigma*(2 + lambda);

        roots_A = roots([a B c]);

        % Keep only real roots with A > 1

        real_roots = roots_A(abs(imag(roots_A)) < 1e-8 & real(roots_A) > 1 + 1e-10);

        if isempty(real_roots)
            error('No physical A equilibrium found.');
        end

        % If there are two physical roots, use the smaller one

        A = min(real(real_roots));

    end

    % Analytical H(t) equilibrium

    ana_H = @(x) (b - d - (alpha + xi)*x) / nu;

    % Analytical L(t) equilibrium

    ana_L = @(A,x) (x/(beta*psi)) .* (b + sigma + alpha*A - xi*x);

    % L governing equilibrium equation

    function value = L_govEq(x)

        A = A_root(x);
        H = ana_H(x);
        L = ana_L(A,x);

        value = h*x*H - delta*L - beta*L*H + r*x*H*(d + nu*H + alpha*(A + x));

        if ~isfinite(value)
            error('Non-finite equilibrium equation.');
        end

    end

    % Find x(t) equilibrium

    x_upper = (b - d)/(alpha + xi);

    x_grid = linspace(1e-7, x_upper*(1 - 1e-10), 500);

    f_values = [];

    for i = 1:length(x_grid)

        x_test = x_grid(i);

        try
            f_test = L_govEq(x_test);

            if isfinite(f_test)
                f_values = [f_values; x_test, f_test];
            end

        catch

            continue
        end

    end

    if size(f_values,1) < 2
        error('Could not find a valid bracket for x equilibrium.');
    end

    % Search for a sign-changing bracket

    x0 = [];

    for i = 1:size(f_values,1)-1

        x_left  = f_values(i,1);
        f_left  = f_values(i,2);

        x_right = f_values(i+1,1);
        f_right = f_values(i+1,2);

        % Exact zero
        if f_left == 0
            x0 = x_left;
            break;
        end

        % Sign change
        if f_left * f_right < 0

            x0 = fzero(@L_govEq, [x_left, x_right]);

            break;
        end

    end

    if isempty(x0)
        error('No endemic x equilibrium found.');
    end

    % Construct equilibrium

    A0 = A_root(x0);
    H0 = ana_H(x0);
    L0 = ana_L(A0,x0);

    equilibrium = [H0; x0; A0; L0];

    if any(~isfinite(equilibrium)) || any(equilibrium <= 0)
        error('Computed endemic initial state is not variable.');
    end

end