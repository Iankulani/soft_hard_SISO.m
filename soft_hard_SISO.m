% soft_hard_SISO.m

clear all, close all;

% Prompt the user for inputs
decision = input('Enter decision mode (0 for hard decision, 1 for soft decision):');

L_packet = input('Enter the packet length (L_packet):');
b = input('Enter the number of bits per symbol (b):');
EbN0dBs = input('Enter the range of Eb/N0 (in dB) as a vector:');

% Constants based on the user's input
K = 1; N = 2; Rc = K / N;
N_symbol = (L_packet * N + 12) / b;
sq05 = sqrt(1 / 2);

% Initialize PER vector
PER = zeros(1, length(EbN0dBs));

% Main loop over Eb/N0 values
for i = 1:length(EbN0dBs)
    EbN0dB = EbN0dBs(i);
    nope = 0; % Number of packet errors
    i_packet = 0;
    
    % Simulate multiple packets for error estimation
    while i_packet < 1000 % Limit to 1000 packets for estimation
        bit_strm = randint(1, L_packet);  % Generate random bit stream
        coded_bits = convolution_encoder(bit_strm);  % Convolutional encoding
        
        % Map coded bits to QAM16 symbols
        symbol_strm = QAM16(coded_bits);
        
        % Simulate the fading channel with complex noise
        h = sq05 * (randn(1, N_symbol) + 1j * randn(1, N_symbol));
        faded_symbol = symbol_strm .* h;  % Apply channel fading
        
        % Calculate the received signal power
        P_b = mean(abs(faded_symbol).^2) / b;
        noise_amp = sqrt(P_b / 2 * 10^(-EbN0dB / 10));
        
        % Add noise to the received signal
        faded_noisy_symbol = faded_symbol + noise_amp * (randn(1, N_symbol) + 1j * randn(1, N_symbol));
        
        % Compensate for the channel effect
        channel_compensated = faded_noisy_symbol ./ h;
        
        % Decision-making based on the selected mode
        if decision == 0
            % Hard decision mode
            sliced_symbol = QAM16_slicer(channel_compensated);
            hard_bits = QAM16_demapper(sliced_symbol);
            Viterbi_init;
            bit_strm_hat = Viterbi_decode(hard_bits);
        else
            % Soft decision mode
            soft_bits = soft_decision_sigma(channel_compensated, h);
            Viterbi_init;
            bit_strm_hat = Viterbi_decode_soft(soft_bits);
        end
        
        % Compare the decoded bits with the transmitted bits
        bit_strm_hat = bit_strm_hat(1:L_packet);  % Ensure the decoded bits are the same length as the original
        nope = nope + sum(bit_strm ~= bit_strm_hat);  % Count packet errors
        
        i_packet = i_packet + 1;
        if nope > 50, break; end  % Stop if too many errors
    end
    
    % Compute the Packet Error Rate (PER)
    PER(i) = nope / i_packet;
    if PER(i) < 1e-2, break; end  % Stop if PER is below threshold
end

% Plot the result
semilogy(EbN0dBs, PER, 'k-o');
xlabel('Eb/N0 [dB]');
ylabel('Packet Error Rate (PER)');
grid on;
xlim([0 EbN0dBs(end)]);
ylim([1e-3 1]);

% Function Definitions (for simulation)

function coded_bits = convolution_encoder(bits)
    % Placeholder for convolutional encoder function
    % For demonstration purposes, simply return the input as coded_bits
    coded_bits = bits;  % In a real case, implement convolution encoding here
end

function symbols = QAM16(bits)
    % Map bits to 16-QAM symbols
    symbols = reshape(bits, 4, length(bits) / 4);
    symbols = bi2de(symbols', 'left-msb');
end

function [sliced_symbols] = QAM16_slicer(received_signal)
    % Placeholder for 16-QAM slicer
    % For demonstration purposes, implement your own slicing logic
    sliced_symbols = received_signal; % Modify to implement slicing logic
end

function demodulated_bits = QAM16_demapper(sliced_symbols)
    % Map 16-QAM symbols back to bits
    demodulated_bits = de2bi(sliced_symbols, 4, 'left-msb');
    demodulated_bits = demodulated_bits';
end

function soft_bits = soft_decision_sigma(channel_compensated, h)
    % Placeholder for soft decision function (soft decision based on received symbols)
    soft_bits = real(channel_compensated .* conj(h));  % Example soft decision function
end

function decoded_bits = Viterbi_decode(hard_bits)
    % Placeholder for Viterbi decoding (hard decision)
    decoded_bits = hard_bits;  % Implement actual Viterbi decoding here
end

function decoded_bits = Viterbi_decode_soft(soft_bits)
    % Placeholder for Viterbi decoding (soft decision)
    decoded_bits = soft_bits;  % Implement actual Viterbi decoding here
end

function [val] = randint(m, n)
    % Generate random binary values between 0 and 1
    val = randi([0 1], m, n);
end
