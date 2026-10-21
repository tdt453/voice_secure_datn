% =========================================================================
% POLAR CRYPTO-CODING: 5G AUDIO SECURITY (POLAR CODES + SNOW-V STREAM CIPHER)
% =========================================================================
clear; clc; close all;

%% 1. THAM SỐ HỆ THỐNG
N = 1024;           % Chiều dài mã Polar
K = 512;            % Số bit mang tin (Information bits)
Kf = N - K;         % Số bit đóng băng (Frozen bits)
EbNo_dB = 0:2:12;   % Dải SNR khảo sát

% Tham số kênh Rician (K = 3dB)
K_factor_dB = 3;    
K_lin = 10^(K_factor_dB/10);
mu_rice = sqrt(K_lin / (2*(K_lin+1)));      
sig_rice = sqrt(1 / (2*(K_lin+1)));         

% Các kịch bản lỗi khóa
err_cases = [0, 1, round(0.25*Kf), round(0.5*Kf), round(0.75*Kf), Kf];
err_labels = {'Correct Key', '1-bit Error', '25% Error', '50% Error', '75% Error', '100% Error'};
plot_styles = {'k-o', 'b-s', 'g-^', 'm-d', 'c-v', 'r-x'};

%% 2. XỬ LÝ ÂM THANH (ĐỌC FILE human.mp3)
fprintf('\n--- 1. ĐANG XỬ LÝ DỮ LIỆU ÂM THANH ---\n');
try
    [audio_data, fs] = audioread('human.mp3');
    % Lấy 0.15s âm thanh đầu tiên để chạy mô phỏng
    y_mono = audio_data(1:min(floor(fs*0.15), length(audio_data)), 1); 
    fprintf('Đã tải thành công file human.mp3.\n');
catch
    warning('Không tìm thấy file. Đang tạo âm thanh giả lập (1kHz tone)...');
    fs = 44100;
    y_mono = sin(2*pi*1000*(0:1/fs:0.15)'); 
end

y_u8 = uint8((y_mono + 1) * 127.5);
y_bits = reshape(dec2bin(y_u8, 8).' - '0', 1, []); 
num_blocks = floor(length(y_bits) / K);            
audio_bits = reshape(y_bits(1:num_blocks * K), K, []).';

fprintf('Số lượng khối dữ liệu (Blocks): %d\n', num_blocks);

% KHỞI TẠO KHÓA BÍ MẬT 5G (MASTER KEY 256-bit cho SNOW-V)
master_key_bytes = randi([0 255], 1, 32, 'uint8'); 

%% 3. KHỞI TẠO MÃ POLAR CHUẨN (BHATTACHARYYA)
fprintf('\n--- 2. TẠO CẤU TRÚC KÊNH POLAR ---\n');
n_log = log2(N); Gn = 1; F = [1 0; 1 1];
for i = 1:n_log, Gn = kron(Gn, F); end

% Tính tham số Bhattacharyya (Z)
Z = zeros(1, N);
Z(1) = 0.5; 
for step = 1:n_log
    len = 2^(step-1);
    for j = 1:len
        Z_prev = Z(j);
        Z(j)       = 2*Z_prev - Z_prev^2;   
        Z(j + len) = Z_prev^2;              
    end
end

% Manual Bit-Reversal
indices = 0:N-1;
bin_strs = dec2bin(indices, n_log);
rev_indices = bin2dec(fliplr(bin_strs)) + 1;
Z_natural = zeros(1, N);
Z_natural(rev_indices) = Z;

% Phân bổ kênh
[~, Reliability] = sort(Z_natural, 'descend'); 
Frozen_Indices = Reliability(1:N-K);    
Info_Indices = Reliability(N-K+1:end);  

is_info = false(1, N);
is_info(Info_Indices) = true;
Seq_Frozen_Indices = find(~is_info); 

%% 4. TIẾN HÀNH MÔ PHỎNG BER
BER_results = zeros(length(err_cases), length(EbNo_dB));
fprintf('\n--- 3. BẮT ĐẦU MÔ PHỎNG VỚI MÃ DÒNG SNOW-V ---\n');

for err_idx = 1:length(err_cases)
    num_err_bits = err_cases(err_idx);
    fprintf('Đang chạy: %-15s | Lỗi %3d bits khóa\n', err_labels{err_idx}, num_err_bits);
    
    for snr_idx = 1:length(EbNo_dB)
        snr = EbNo_dB(snr_idx);
        snr_lin = 10^(snr/10);
        N0 = 1 / (snr_lin * (K/N)); 
        
        total_errors = 0;
        total_bits = 0;
        
        for blk = 1:num_blocks
            % --- A. BÊN PHÁT (SỬ DỤNG SNOW-V STREAM CIPHER) ---
            % Tạo IV 128-bit (16 bytes) cho từng khối
            iv_bytes = zeros(1, 16, 'uint8');
            blk_bytes = typecast(uint32(blk), 'uint8');
            iv_bytes(1:4) = blk_bytes;
            
            % Gọi hàm sinh Keystream từ mô hình kiến trúc SNOW-V
            tx_frozen_key = generate_snow_v_model(master_key_bytes, iv_bytes, Kf);
            
            tx_info = audio_bits(blk, :);
            u = zeros(1, N);
            u(Info_Indices) = tx_info;
            u(Seq_Frozen_Indices) = tx_frozen_key; 
            
            x_enc = mod(u * Gn, 2);
            s_tx = 1 - 2 * x_enc; 
            
            % --- B. KÊNH TRUYỀN RICE ---
            h = (mu_rice + sig_rice*randn(1, N)) + 1i*(mu_rice + sig_rice*randn(1, N));
            h_mag = abs(h);
            noise = sqrt(N0/2) * randn(1, N);
            y_rx = h_mag .* s_tx + noise;
            
            LLR_rx = (4 * y_rx .* h_mag) / N0;
            
            % --- C. BÊN NHẬN ---
            rx_frozen_key = tx_frozen_key;
            if num_err_bits > 0
                err_pos = randperm(Kf, num_err_bits);
                rx_frozen_key(err_pos) = bitxor(rx_frozen_key(err_pos), 1);
            end
            
            rx_u_hat = run_polar_sc_decoder(LLR_rx, is_info, rx_frozen_key);
            
            rx_info = rx_u_hat(Info_Indices);
            total_errors = total_errors + sum(tx_info ~= rx_info);
            total_bits = total_bits + K;
        end
        
        current_ber = total_errors / total_bits;
        if current_ber == 0
            BER_results(err_idx, snr_idx) = 1e-5; % Kéo dài đồ thị thác nước
        else
            BER_results(err_idx, snr_idx) = current_ber;
        end
    end
end
fprintf('Hoàn tất!\n\n');

%% 5. VẼ BIỂU ĐỒ KẾT QUẢ
figure('Color', 'w', 'Name', '5G Audio Security: Polar + SNOW-V', 'Position', [150 150 800 600]);
hold on; grid on;

for i = 1:length(err_cases)
    semilogy(EbNo_dB, BER_results(i, :), plot_styles{i}, 'LineWidth', 2, 'MarkerSize', 8);
end

set(gca, 'YScale', 'log');
axis([min(EbNo_dB) max(EbNo_dB) 1e-5 1.5]); 
xlabel('Eb/N0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Bit Error Rate (BER) of Audio Data', 'FontSize', 12, 'FontWeight', 'bold');
title('Anti-Eavesdropping Performance (Polar Codes + SNOW-V Model)', 'FontSize', 14);
legend(err_labels, 'Location', 'southwest', 'FontSize', 11);
set(gca, 'FontSize', 11);

%% ================= HÀM MÔ PHỎNG KIẾN TRÚC SNOW-V =================
function keystream = generate_snow_v_model(key_bytes, iv_bytes, num_bits)
    % Mô hình kiến trúc SNOW-V: 256-bit Key, 128-bit IV
    % Sử dụng mảng uint32 (4 phần tử = 128 bit) để tăng tốc độ trên MATLAB
    
    K = typecast(uint8(key_bytes), 'uint32'); % 8 words = 256 bit
    IV = typecast(uint8(iv_bytes), 'uint32'); % 4 words = 128 bit
    
    % 1. Khởi tạo trạng thái (Initialization)
    % LFSR: A, B (128-bit mỗi thanh ghi)
    A = bitxor(K(1:4), IV);
    B = K(5:8);
    
    % FSM: R1, R2, R3 (128-bit mỗi thanh ghi)
    R1 = zeros(1, 4, 'uint32');
    R2 = zeros(1, 4, 'uint32');
    R3 = zeros(1, 4, 'uint32');
    
    % 2. Chạy Pre-clocking (Mô phỏng FSM Update)
    for step = 1:16
        % Cập nhật FSM (Sử dụng phép toán phi tuyến cơ bản thay cho AES-round)
        T1 = bitxor(R1, R2);
        T2 = R3;
        R3 = R2;
        R2 = R1;
        R1 = T1 + T2; % Cộng modulo 2^32
        R1 = bitxor(R1, bitshift(R1, 7)); % Khếch đại khuếch tán (Diffusion)
        
        % Cập nhật LFSR
        next_A = bitxor(B, bitshift(A, 3));
        A = B;
        B = next_A;
    end
    
    % 3. Sinh dòng khóa (Keystream Generation)
    num_clocks = ceil(num_bits / 128); % Mỗi xung nhịp xuất ra 128 bit
    ks_words = zeros(1, num_clocks * 4, 'uint32');
    
    for clk = 1:num_clocks
        % Công thức sinh dòng khóa SNOW-V: z = (R1 [+] R2) XOR A
        z = bitxor(R1 + R2, A);
        ks_words((clk-1)*4 + 1 : clk*4) = z;
        
        % Cập nhật FSM và LFSR cho chu kỳ tiếp theo
        T1 = bitxor(R1, R2);
        T2 = R3;
        R3 = R2;
        R2 = R1;
        R1 = T1 + T2;
        R1 = bitxor(R1, bitshift(R1, 7));
        
        next_A = bitxor(B, bitshift(A, 3));
        A = B;
        B = next_A;
    end
    
    % 4. Chuyển đổi thành chuỗi bit
    ks_bytes = typecast(ks_words, 'uint8');
    bit_str = dec2bin(ks_bytes, 8)';
    keystream_full = bit_str(:)' - '0';
    
    % Cắt chuẩn xác theo độ dài yêu cầu (Kf bits)
    keystream = keystream_full(1:num_bits);
end

%% ================= HÀM GIẢI MÃ POLAR SC =================
function u_est = run_polar_sc_decoder(LLR, is_info, frozen_key)
    N_len = length(LLR);
    u_est = zeros(1, N_len);
    f_idx = 1;     
    bit_idx = 1;   
    
    recursive_sc(LLR);
    
    function recursive_sc(L)
        len = length(L);
        if len == 1
            if is_info(bit_idx)
                u_est(bit_idx) = L < 0; 
            else
                u_est(bit_idx) = frozen_key(f_idx); 
                f_idx = f_idx + 1;
            end
            bit_idx = bit_idx + 1;
            return;
        end
        
        half = len / 2;
        L1 = L(1:half);
        L2 = L(half+1:end);
        
        L_left = sign(L1) .* sign(L2) .* min(abs(L1), abs(L2));
        recursive_sc(L_left);
        
        u_left = u_est(bit_idx - half : bit_idx - 1);
        beta = calc_partial_sum(u_left); 
        
        L_right = L2 + (1 - 2 * beta) .* L1;
        recursive_sc(L_right);
    end

    function beta_out = calc_partial_sum(u)
        len_u = length(u);
        if len_u == 1
            beta_out = u;
            return;
        end
        h = len_u / 2;
        b1 = calc_partial_sum(u(1:h));
        b2 = calc_partial_sum(u(h+1:end));
        beta_out = [mod(b1 + b2, 2), b2];
    end
end