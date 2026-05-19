infile = 'music.wav';    
maxSamplesToShow = 62500; 

info = audioinfo(infile);
y_native = audioread(infile, 'native');  
[nSamples, nChannels] = size(y_native);

bitsPerSample = info.BitsPerSample;   

chan = 1;
vec = y_native(:, chan);

N = min(nSamples, maxSamplesToShow);
vec = vec(1:N);

switch class(vec)
    case 'int16'
        uvec = typecast(vec, 'uint16');
    case 'int32'
        uvec = typecast(vec, 'uint32');
    case 'int64'
        uvec = typecast(vec, 'uint64');
    case 'int8'
        uvec = typecast(vec, 'uint8');
    case 'uint8'
        uvec = vec;
    case 'uint16'
        uvec = vec;
    case 'uint32'
        uvec = vec;
    case 'single'  
        uvec = typecast(int16(max(min(vec,1),-1) * (2^15-1)), 'uint16');
        bitsPerSample = 16;
    case 'double'
        uvec = typecast(int16(max(min(vec,1),-1) * (2^15-1)), 'uint16');
        bitsPerSample = 16;
    otherwise
        error('Unsupported sample class: %s', class(vec));
end

bitStrings = dec2bin(uvec, bitsPerSample);  % char matrix (N x bits)

disp('First 5 sample bitstrings (MSB ... LSB):');
disp(bitStrings(1:min(5,size(bitStrings,1)), :));

bitstream_chars = reshape(bitStrings.', 1, []);
bits_col        = double(bitstream_chars - '0').';

% Output sizes
fprintf('Converted %d samples (channel %d) to %d-bit strings.\n', N, chan, bitsPerSample);

L_sim   = 1e6;
bits_tx = repmat(bits_col, ceil(L_sim / numel(bits_col)), 1);
bits_tx = bits_tx(1:L_sim);

%simulation
[EbN0_dB, BER, bits_hat] = adaptive_QAM_sim('rayleigh', true, bits_tx);


bits_hat_matrix = reshape(bits_hat(1 : N*bitsPerSample), bitsPerSample, N).';
bitStr_hat = char(bits_hat_matrix + '0');
uvec_hat = uint16(bin2dec(bitStr_hat));
vec_hat  = typecast(uvec_hat, 'int16');
audiowrite('music_received.wav', vec_hat, info.SampleRate);


%Main Function
function [EbN0_dB, BER, bits_hat] = adaptive_QAM_sim(channel_type, show_plot, bits)

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

    z_last = [];
    M_last = 4;

    for i = 1:length(EbN0_dB)
        % Adaptive modulation selection
        if EbN0_dB(i) < 8
            M = 4;      % QPSK
        else
            M = 16;     % 16-QAM
        end
        M_last = M;

        k = log2(M);
        m = sqrt(M);
        Rb = Rs * k;
        BW = Rs;
        d = 2;

        levels = (-(m-1):2:(m-1)) * d/2;

        % Average energy
        Eavg =  mean(levels.^2) * d^2 / 2;
        Eb = Eavg/k;

        
        % Generate bits

        Nsym = L / k;
        bit_matrix = reshape(bits(1:L), k, Nsym).';

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
                h_sym = ones(1, Nsym);
                
            case 'rayleigh'
                h_sym = (1/sqrt(2)) * (randn(1, Nsym) + 1j*randn(1, Nsym));
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
        z = z(delay+1 : f_ovsamp : delay + Nsym*f_ovsamp);
        z = z / f_ovsamp;
        
        %Equalization
        z = z ./ h_sym;

        
        % Detection
        I_hat = zeros(1, Nsym);
        Q_hat = zeros(1, Nsym);

        for n = 1:Nsym

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

        BER(i) = mean(bits(1:L) ~= bits_hat);

        % Print results including rates and bandwidth
        fprintf('SNR = %2d dB | M = %2d | Rb = %4.1f Mbps | Rs = %3.1f Msps | BW = %3.1f MHz | BER = %e\n', ...
            EbN0_dB(i), M, Rb/1e6, Rs/1e6, BW/1e6, BER(i));

        z_last = z;
    end

    % ==============================
    % BER plot
    % ==============================

    if show_plot
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
        Nplot = min(5000, length(z_last));
        plot(real(z_last(1:Nplot)), imag(z_last(1:Nplot)), '.');
        grid on;
        axis([-5 5 -5 5]); % Sets the fixed limit layout from -5 to 5
        xlabel('In-Phase');
        ylabel('Quadrature');
        title('Received Constellation');
    end
end

function b = gray2bin(g)
    b = g;
    shift = floor(g/2);

    while any(shift(:))
        b = bitxor(b, shift);
        shift = floor(shift/2);
    end
end