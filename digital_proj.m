function [EbN0_dB, BER] = adaptive_QAM_sim(channel_type, show_plot)

    clc;

    % Parameters
    L = 1e6;
    f_ovsamp = 8;
    Rs = 6.5e6;   
    EbN0_dB = 0:2:20;
    EbN0 = 10.^(EbN0_dB/10);

    BER = zeros(size(EbN0_dB));

    rect = ones(1,f_ovsamp);

    gray = @(x) bitxor(x, floor(x/2));

    for i = 1:length(EbN0_dB)
        % Adaptive modulation selection
        if EbN0_dB(i) < 8
            M = 4;      % QPSK
        else
            M = 16;     % 16-QAM
        end

        k = log2(M);
        m = sqrt(M);
        Rb = Rs * k;
        BW = Rs;
        d = 2;

        levels = (-(m-1):2:(m-1)) * d/2;

        % Average energy
        Eavg = mean(levels.^2) * d^2 / 2;
        Eb = Eavg/k;

        
        % Generate bits

        bits = randi([0 1], L*k, 1);

        bit_matrix = reshape(bits, k, []).';
        sym_idx = bi2de(bit_matrix, 'left-msb');

        I_bin = mod(sym_idx, m);
        Q_bin = floor(sym_idx/m);

        I_gray = gray(I_bin);
        Q_gray = gray(Q_bin);

        I = levels(I_gray + 1);
        Q = levels(Q_gray + 1);

        sk = I + 1j*Q;
            
        
        % Pulse shaping
        

        s_up = upsample(sk, f_ovsamp);

        x = conv(s_up, rect);

        
        % Noise
        

        sigma = sqrt(f_ovsamp*Eb/(2*EbN0(i)));

        noise = sigma *(randn(size(x)) + 1j*randn(size(x)));

        % ==============================
        % Channel
        % ==============================

        switch lower(channel_type)
            case 'awgn'
                r = x + noise;
                h_sym = ones(1, L);
                
            case 'rayleigh'
                h_sym = (1/sqrt(2)) * (randn(1, L) + 1j*randn(1, L));
                h_full = repelem(h_sym, f_ovsamp);
                % 3. Pad the end to account for convolution tail length of x
                h_full = [h_full, h_full(end)*ones(1, length(x) - length(h_full))];
                
                % Apply fading and add noise
                r = h_full .* x + noise;
                
            otherwise
                error('Unknown channel');
        end   
     
        % Matched Filter
     
        z = conv(r, rect);
        delay = f_ovsamp - 1;
        z = z(delay+1 : f_ovsamp : delay + L*f_ovsamp);
        z = z / f_ovsamp;
        
        %Equalization
        z = z ./ h_sym;

        
        % Detection
        I_hat = zeros(1,L);
        Q_hat = zeros(1,L);

        for n = 1:L

            [~, idxI] = min(abs(real(z(n)) - levels));

            [~, idxQ] = min(abs(imag(z(n)) - levels));

            I_hat(n) = idxI - 1;
            Q_hat(n) = idxQ - 1;

        end

        
        % Gray decoding

        I_bin_hat = gray2bin(I_hat);
        Q_bin_hat = gray2bin(Q_hat);

        sym_hat = Q_bin_hat * m + I_bin_hat;

        bits_hat = de2bi(sym_hat, k, 'left-msb');

        bits_hat = bits_hat.';
        bits_hat = bits_hat(:);

        BER(i) = mean(bits ~= bits_hat);

        % Print results including rates and bandwidth
        fprintf('SNR = %2d dB | M = %2d | Rb = %4.1f Mbps | Rs = %3.1f Msps | BW = %3.1f MHz | BER = %e\n', ...
            EbN0_dB(i), M, Rb/1e6, Rs/1e6, BW/1e6, BER(i));

    end

    % ==============================
    % BER plot
    % ==============================

    figure;

    semilogy(EbN0_dB, BER, 'bo-','LineWidth',2);

    grid on;

    xlabel('E_b/N_0 (dB)');
    ylabel('BER');

    title(['Adaptive QAM over ' upper(channel_type)]);

    % ==============================
    % Final constellation
    % ==============================

    figure;

    plot(real(z(1:5000)), imag(z(1:5000)), '.');

    grid on;

    title('Received Constellation');

end
function b = gray2bin(g)
    b = g;
    shift = floor(g/2);

    while any(shift(:))
        b = bitxor(b, shift);
        shift = floor(shift/2);
    end
end


adaptive_QAM_sim('rayleigh',true);