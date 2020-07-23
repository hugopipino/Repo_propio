% Ejercicio 1
clear; close all; clc

%% Apartado A
s = tf('s');
% Planta
Gs = (20*s+500)/((3*s+1)*(5*s+1))
% Periodo de muestreo
Ts = 0.1;
% Discretizacion
Gz = c2d(Gs, Ts, 'zoh')

%% Apartado B y C
% Create an input_prbs_signal
samples = 1200;
band = [0 0.05];
range = [-1, 1];
noise_amplitude = 0.1;
input_prbs_signal = idinput(samples, 'PRBS', band, range);

% Simulate output and add noise
simulated_output = sim(input_prbs_signal, idpoly(Gz));
simulated_noised_output = add_white_noise_to_func(simulated_output, noise_amplitude);

% Armado del paquete de identificacion
ident_proportion = 0.5;  % 50 percent for identification
plot_package = true;
[data_ident, data_test] = generate_ident_package(input_prbs_signal, simulated_noised_output, Ts, ident_proportion, plot_package);

%% Identifico con la herramienta de estimacion de matlab
na = 2; nb = 2; nk = 1;
focus_mode = 'prediction';
residual_analysis = true;
Gzi = discrete_ident_arx(data_ident, Ts, focus_mode, na, nb, nk, residual_analysis);

%% Identifico por minimos cuadrados en forma recursiva
plot_ident = true;
Gzi_mc = discrete_ident_recursive_least_squares(data_ident, Ts, plot_ident)

%% Validacion de resultados
validate_identifications(data_test, Gzi, Gzi_mc)


function noisy = add_white_noise_to_func(clean_signal, noise_amplitude)
	%#ADD_WHITE_NOISE_TO_FUNC agrega ruido blanco a una señal
	%#
	%# SYNOPSIS add_white_noise_to_func(clean_signal, noise_amplitude)
	%# INPUT clean_signal: (simbólico) la señal de entrada
	%# INPUT noise_amplitude: (float) amplitud de la señal de ruido
	%# OUTPUT noisy (simbólico) señal con ruido agregado
	%#
    noise = noise_amplitude * randn(size(clean_signal));
    noisy = clean_signal + noise;
end

function [data_ident, data_validation] = generate_ident_package(input_signal, output_signal, sample_time, ident_proportion, plot_package)
	%#GENERATE_IDENT_PACKAGE arma e imprime el paquete de datos
	%#
	%# SYNOPSIS generate_ident_package(input_signal, output_signal, sample_time, ident_proportion, plot_package)
	%# INPUT input_signal: (simbólico) la señal de entrada
	%# INPUT output_signal (simbólico) la señal de salida
	%# INPUT sample_time (float) tiempo de muestreo
	%# OUTPUT [data_ident(paquete), data_validation(paquete)]
    data_size = size(input_signal);
    ident_size = floor(data_size * ident_proportion);
    data = iddata(output_signal, input_signal, sample_time);
    data_ident = data(1:ident_size);
    data_validation = data(ident_size+1:data_size);
    
    if plot_package
        figure(1)
        set(data, 'InputName', 'Entrada', 'OutputName', 'Salida')
        plot(data(1:ident_size), 'b', data(ident_size:data_size), 'g');
        legend('Identificación', 'Validación');
        xlabel('Tiempo [s]'); ylabel(''); title('');
        print -dsvg G1EJ1img1.svg
    end
end

function Gzi = discrete_ident_arx(data, Ts, focus_mode, na, nb, nk, residual_analysis)
    %#DISCRETE_IDENT_ARX arma e imprime el paquete de datos
	%#
	%# SYNOPSIS discrete_ident_arx(data, Ts, focus_mode, na, nb, nk, residual_analysis)
	%# INPUT data(package): 
	%# INPUT Ts(float):
	%# INPUT focus_mode(string):
	%# INPUT na(float):
	%# INPUT nb(float):
	%# INPUT nk(float):
	%# INPUT residual_analysis(boolean):
	%# OUTPUT Gzi(tf):
    Opt = arxOptions;                     
    Opt.Focus = focus_mode;   
    sys_id = arx(data, [na nb nk], Opt);
    [num, den] = tfdata(sys_id);
    Gzi = tf(num, den, Ts)

    frecuency_sampling = 1/Ts;
    if residual_analysis
        analyze_residuals(data, sys_id, frecuency_sampling)
    end
end

function Gzi_mc = discrete_ident_recursive_least_squares(data, Ts, plot_ident)
	%#DISCRETE_IDENT_RECURSIVE_LEAST_SQUARES método de los minimos cuadrados
	%#
	%# SYNOPSIS discrete_ident_recursive_least_squares(data, Ts, plot_ident)
	%# INPUT data(paquete): 
	%# INPUT Ts(float):
	%# INPUT plot_ident(boolean):
	%# OUTPUT Gzi_mc:
    data_size = size(data, 1);
    n = 3;
    u = data.InputData;
    y = data.OutputData;
    Theta = zeros(4, data_size);
    P = 1e12*eye(4);

    for k = n:data_size-1
        Phi = [-y(k-1) -y(k-2) u(k-1) u(k-2)];
        K = P*Phi'/(1+(Phi*P*Phi')); 
        Theta(:,k+1) = Theta(:,k)+K*(y(k)-Phi*Theta(:,k));
        P = P-(K*Phi*P);
    end
    Gzi_mc = tf(Theta(3:4,data_size)', [1 Theta(1:2,data_size)'], Ts)

    if plot_ident
        figure(2);
        subplot(2, 2, 1); plot(Theta(1, :));xlabel('Tiempo [s]'); grid
        set(gca, 'XTickLabel', 0:10:data_size); ylabel('a_1')
        subplot(2, 2, 2); plot(Theta(2, :)); xlabel('Tiempo [s]');grid
        set(gca, 'XTickLabel', 0:10:data_size); ylabel('a_2')
        subplot(2, 2, 3); plot(Theta(3, :)); xlabel('Tiempo [s]');grid
        set(gca, 'XTickLabel', 0:10:data_size); ylabel('b_1')
        subplot(2, 2, 4); plot(Theta(4, :)); grid
        set(gca, 'XTickLabel', 0:10:data_size); ylabel('b_2')
        xlabel('Tiempo [s]');
        print -dsvg G1EJ1img2.svgend
    end
end

function analyze_residuals(data, sys_id, sampling_frequency)
    %#ANALYZE_RESIDUALS análisis de residuos
	%#
	%# SYNOPSIS analyze_residuals(data, sys_id, sampling_frequency)
	%# INPUT data(paquete): 
	%# INPUT sys_id( - ):
	%# INPUT sampling_frequency(float):
    %% Analisis de residuos
    %% Analisis de residuos
    e = resid(sys_id, data);

    figure(4); 
    % Auto-correlación del residuo
    subplot(2, 2, 1); 
    [Rmm, lags] = xcorr(e.y, 'coeff');
    Rmm = Rmm(lags>0); lags = lags(lags>0);
    plot(lags/sampling_frequency,Rmm); xlabel('Lag [s]');
    title('Auto-corr. residuo');

    % Correlación del residuo con la salida
    subplot(2, 2, 2); 
    [Rmm, lags] = xcorr(e.y, data.OutputData, 'coeff');
    Rmm = Rmm(lags>0); lags = lags(lags>0);
    plot(lags/sampling_frequency, Rmm); xlabel('Lag [s]');
    title('Corr. residuo/salida');

    % Histograma del residuo
    subplot(2, 2, 3); 
    histfit(e.y); title('Histograma residuo');

    % Correlación del residuo con la salida
    subplot(2, 2, 4); 
    [Rmm, lags] = xcorr(e.y, data.InputData, 'coeff');
    Rmm = Rmm(lags>0); lags = lags(lags>0);
    plot(lags/sampling_frequency, Rmm); xlabel('Lag [s]');
    title('Corr. residuo/entrada');
    print -dsvg G1EJ1img4.svg
end

function validate_identifications(data, Gzi, Gzi_mc)
    %#VALIDATE_IDENTIFICATIONS 
	%#
	%# SYNOPSIS validate_identifications(data, Gzi, Gzi_mc)
	%# INPUT data(paquete): 
	%# INPUT Gzi(tf):
	%# INPUT Gzi_mc(tf):
    data_size = size(data);
    [y_sys, fit] = compare(data, Gzi);
    [y_mc, fit_mc] = compare(data, Gzi_mc);
    t = (1:data_size);
    
    figure(3);
    plot(t, y_sys.OutputData, 'r', t, y_mc.OutputData, 'g--', t, data.OutputData, 'b-.');
    title('Validación de resultados');
    set(gca, 'XTickLabel', 60:10:data_size);
    xlabel('Tiempo [s]');
    legend(sprintf('ARX (%2.2f)', fit), sprintf('RLS (%2.2f)', fit_mc), 'Salida', 'Location', 'SouthEast');
    print -dsvg G1EJ1img3.svg
end
