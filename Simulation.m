clc; clear; close all;

%% 1. Data Generation and Parameter Initialization
% Simulate data (mimicking data2025.mat)
T = 600;
delta_t = 0.005;
N_total = round(T / delta_t); % 600 / 0.005 = 120000
omiga = 4;
zeta = 0.01;
S_f0 = 1;
sigma_e = sqrt(0.01); % Adjusted to match sigma_e^2 ~ 0.01

% Generate synthetic data
t = (0:N_total-1) * delta_t; % t = 0:0.005:599.995
f = sqrt((2 * pi * S_f0) / delta_t) * randn(1, N_total);
y = zeros(2, N_total);
A = [0 1; -omiga^2 -2 * omiga * zeta];
B = [0 1]';
A_d = expm(A * delta_t);
B_d = inv(A) * (A_d - eye(size(A))) * B;
for n = 1:N_total-1
    y(:, n+1) = A_d * y(:, n) + B_d * f(n);
end
z = y(1, :) + sigma_e * randn(1, N_total);

% Mimic data structure
data.measurement = z;
data.T = T;
data.delta_t = delta_t;

% Variable name compatibility
z = get_field(data, 'measurement', 'z');
T = get_field(data, 'T', 600);
delta_t = get_field(data, 'delta_t', 0.005);

%% 2. Data Segmentation
time_start = 0;
time_end = 600;
start_idx = max(1, round(time_start/delta_t));
end_idx = min(length(z), round(time_end/delta_t));
z_seg = z(start_idx:end_idx);
T_seg = (end_idx - start_idx + 1) * delta_t;
N = length(z_seg);

fprintf('Analysis points: %d, Sampling frequency: %.2f Hz\n', N, 1/delta_t);

%% 3. Spectrum Estimation
k = 1500;
w_arr = (2 * pi * (1:k)) / T_seg;
spec_tmp = compute_spectrum(z_seg, w_arr, delta_t, N);
S_meas = (delta_t / (2 * pi * N)) * abs(spec_tmp).^2;

%% 4. Spectrum Smoothing
smooth_win = 15;
S_meas_smooth = movmean(S_meas, smooth_win);

%% 5. Main Peak Detection
[~, pk_idx] = max(S_meas_smooth(1:k));
target_val = 0.8 * S_meas_smooth(pk_idx);
idx_80 = find(S_meas_smooth(1:pk_idx) >= target_val, 1, 'first');
if isempty(idx_80), idx_80 = pk_idx; end
omega_init = w_arr(idx_80);

figure;
semilogy(w_arr, S_meas_smooth(1:k), 'Color', [0.3 0.5 0.7], 'LineWidth', 1.5); hold on;
plot(omega_init, S_meas_smooth(idx_80), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\omega (rad/s)'); ylabel('S_{meas}(\omega)');
title('Smoothed Spectrum');
grid on; set(gca, 'FontSize', 12);

%% 6. Parameter Estimation
x0 = [4.0, 0.01, S_meas_smooth(idx_80), 0.01]; % Adjusted initial guess
lb = [3.8, 0.001, 1e-6, 1e-5];
ub = [4.2, 0.99, 1e3, 1];
fun = @(x) fun_BSDA_weighted(x, N, delta_t, T_seg, k, w_arr, S_meas);

options = optimoptions('fmincon', 'Display', 'off', ...
    'MaxFunctionEvaluations', 20000, 'MaxIterations', 10000, ...
    'TolFun', 1e-8, 'TolX', 1e-8);
x_opt = fmincon(fun, x0, [], [], [], [], lb, ub, [], options);

disp('Parameter Estimation Results:');
fprintf('Natural frequency ω: %.4f rad/s (%.4f Hz)\n', x_opt(1), x_opt(1)/(2*pi));
fprintf('Damping ratio ζ: %.4f\n', x_opt(2));
fprintf('White noise spectrum S_f0: %.4f\n', x_opt(3));
fprintf('Measurement noise σ_e^2: %.4f\n', x_opt(4));

%% 7. Uncertainty Analysis
theta_n = 4;
H = compute_hessian(x_opt, theta_n, fun, N, delta_t, T_seg, k, w_arr, S_meas);
sigma = compute_covariance(H);
SD = sqrt(abs(diag(sigma)));
COV = SD ./ abs(x_opt(:));

% Define actual values
actual_values = [omiga; zeta; S_f0; sigma_e^2];

% Compute Normalized Difference (ND)
ND = abs(x_opt(:) - actual_values) ./ SD;

% Create table
disp('Table 1 Simulation test optimization results');
fprintf('-----------------------------------------------------\n');
fprintf('| Parameter           | Actual  | Optimal | SD    | COV   | ND    |\n');
fprintf('-----------------------------------------------------\n');
fprintf('| Natural frequency (ω) | %.4f | %.4f | %.4f | %.4f | %.4f |\n', actual_values(1), x_opt(1), SD(1), COV(1), ND(1));
fprintf('| Damping ratio (ζ)   | %.4f | %.4f | %.4f | %.4f | %.4f |\n', actual_values(2), x_opt(2), SD(2), COV(2), ND(2));
fprintf('| White noise (S_f0)  | %.4f | %.4f | %.4f | %.4f | %.4f |\n', actual_values(3), x_opt(3), SD(3), COV(3), ND(3));
fprintf('| Noise var (σ_e^2)   | %.4f | %.4f | %.4f | %.4f | %.4f |\n', actual_values(4), x_opt(4), SD(4), COV(4), ND(4));
fprintf('-----------------------------------------------------\n');

%% 8. Theoretical vs Measured Spectrum
E_theoretical = compute_theoretical_spectrum(x_opt, N, delta_t, k);
figure;
semilogy(w_arr, S_meas_smooth(1:k), 'Color', [0.3 0.5 0.7], 'LineWidth', 1.5); hold on;
semilogy(w_arr, E_theoretical(2:k+1), 'r--', 'LineWidth', 1.5);
xlabel('\omega (rad/s)'); ylabel('S_{meas}(\omega)');
legend('Measured', 'Theoretical');
title('Measured vs Theoretical Spectrum');
grid on; set(gca, 'FontSize', 12);

%% 9. Gaussian Distribution Plots
param_names = {'\omega (rad/s)', '\zeta', 'S_{f0}', '\sigma_e^2'};
for i = 1:4
    mu = x_opt(i); sigma_val = SD(i);
    x_range = linspace(mu - 4*sigma_val, mu + 4*sigma_val, 200);
    y_gauss = normpdf(x_range, mu, sigma_val);
    figure;
    plot(x_range, y_gauss, 'Color', [0.3 0.5 0.7], 'LineWidth', 2); hold on;
    xlabel(param_names{i}); ylabel('Probability Density');
    title(['Gaussian Distribution of ', param_names{i}]);
    grid on; legend('Gaussian PDF'); set(gca, 'FontSize', 12);
end

%% Helper Functions
function val = get_field(data, field1, default)
    if isfield(data, field1)
        val = data.(field1);
    elseif isfield(data, 'z')
        val = data.z;
    else
        if isnumeric(default)
            val = default;
        else
            error('Please check data variable name');
        end
    end
end

function spec_tmp = compute_spectrum(z_seg, w_arr, delta_t, N)
    n = (0:N-1)';
    spec_tmp = sum(z_seg(:) .* exp(-1i * w_arr .* n * delta_t), 1);
end

function H = compute_hessian(x_opt, theta_n, fun, N, delta_t, T_seg, k, w_arr, S_meas)
    H = zeros(theta_n);
    dtheta = 1e-4;
    for n = 1:theta_n
        dTheta = zeros(1, theta_n); dTheta(n) = dtheta;
        y1 = x_opt + dTheta; y2 = x_opt - dTheta;
        y1(4) = max(y1(4), 0.001); y2(4) = max(y2(4), 0.001);
        H(n,n) = (fun(y1) - 2*fun(x_opt) + fun(y2)) / (dtheta^2);
        for m = 1:theta_n
            if n == m, continue; end
            dTheta_m = zeros(1, theta_n); dTheta_m(m) = dtheta;
            temp1 = x_opt + dTheta + dTheta_m;
            temp2 = x_opt + dTheta - dTheta_m;
            temp3 = x_opt - dTheta + dTheta_m;
            temp4 = x_opt - dTheta - dTheta_m;
            temp1(4) = max(temp1(4), 0.001); temp2(4) = max(temp2(4), 0.001);
            temp3(4) = max(temp3(4), 0.001); temp4(4) = max(temp4(4), 0.001);
            H(n,m) = (fun(temp1) - fun(temp2) - fun(temp3) + fun(temp4)) / (4*dtheta^2);
        end
    end
end

function sigma = compute_covariance(H)
    if rcond(H) < eps || any(isnan(H(:))) || any(isinf(H(:)))
        sigma = pinv(H + 1e-10 * eye(size(H)));
    else
        sigma = inv(H);
    end
end

function E_theoretical = compute_theoretical_spectrum(x_opt, N, delta_t, k)
    [omiga_opt, zeta_opt, S_f0_opt, sigma_e_opt] = deal(x_opt(1), x_opt(2), x_opt(3), x_opt(4));
    R = zeros(1, N);
    if zeta_opt == 0
        tau = (1:N-1) * delta_t;
        R(1:N-1) = cos(omiga_opt * tau) * pi * S_f0_opt / (2.0 * omiga_opt^3);
    else
        Omiga_d = sqrt(1 - zeta_opt^2) * omiga_opt;
        tau = (1:N-1) * delta_t;
        R(1:N-1) = (cos(Omiga_d * tau) + sin(Omiga_d * tau) * zeta_opt / sqrt(1 - zeta_opt^2)) .* ...
                   exp(-zeta_opt * omiga_opt * tau) * pi * S_f0_opt / (2.0 * omiga_opt^3 * zeta_opt);
    end
    temp3 = (sigma_e_opt^2 * delta_t) / (2 * pi);
    temp4 = delta_t / (2 * pi * N);
    En = real(fft([N, 2*(N-1:-1:2).*R(2:N-1)]));
    E_theoretical = temp4 * En + temp3 + 1e-8;
end

%% Weighted Objective Function
function output = fun_BSDA_weighted(input, N, delta_t, T, k, w_k, S_yn)
    omiga = input(1); zeta = input(2); S_f0 = input(3); sigma_e = input(4);
    if any(input < 0) || zeta >= 1 || sigma_e < 0.0001 || omiga < 0.1
        output = Inf; return;
    end
    if zeta > 0.95, zeta = 0.95; end
    R = zeros(1, N);
    if zeta == 0
        tau = (1:N-1) * delta_t;
        R(1:N-1) = cos(omiga * tau) * pi * S_f0 / (2.0 * omiga^3);
    else
        Omiga_d = sqrt(1 - zeta^2) * omiga;
        tau = (1:N-1) * delta_t;
        R(1:N-1) = (cos(Omiga_d * tau) + sin(Omiga_d * tau) * zeta / sqrt(1 - zeta^2)) .* ...
                   exp(-zeta * omiga * tau) * pi * S_f0 / (2.0 * omiga^3 * zeta);
        R(isnan(R) | isinf(R)) = 0;
    end
    temp3 = (sigma_e^2 * delta_t) / (2 * pi);
    temp4 = delta_t / (2 * pi * N);
    En = real(fft([N, 2*(N-1:-1:2).*R(2:N-1)]));
    E_initial = temp4 * En + temp3 + 1e-8;
    E_initial(E_initial <= 0) = 1e-10;
    weight = linspace(1, 0.2, k);
    Result = sum(weight .* (S_yn(1:k) ./ E_initial(2:k+1) + log(E_initial(2:k+1))));
    output = Result;
end