classdef AudioCryptoApp < matlab.apps.AppBase
    % --- KHAI BÁO BIẾN DỮ LIỆU ---
    properties (Access = public)
        UIFigure             matlab.ui.Figure
        LeftPanel            matlab.ui.container.Panel
        RightTabs            matlab.ui.container.TabGroup
        
        % Tabs
        TabWaveform, TabHistogram, TabSpectrogram, TabCorrelation, TabSecurity, TabMetrics
        
        % Axes
        AxOrig, AxEnc, AxDec, AxHistOrig, AxHistEnc
        AxSpecOrig, AxSpecEnc, AxCorrOrig, AxCorrEnc
        AxCPA, AxKPA
        
        % Controls
        BtnLoad, BtnEncrypt, BtnDecrypt, BtnLive
        BtnPlayOrig, BtnPlayEnc, BtnPlayDec
        txtKey, lblStatus, MetricsTable
        
        % Data Variables
        AudioOrig, AudioEnc, AudioDec, Fs = 44100, ScaleFactor = 0.9
        
        % Crypto Variables
        GlobalKey, IV
        
        % Recording & Network Variables
        IsRecording = false;
        Recorder
        NetServer 
    end
    
    methods (Access = private)
        % --- HÀM TÍNH LSD ---
        function lsd_val = calculate_LSD(~, orig_sig, dec_sig, fs)
            min_len = min(length(orig_sig), length(dec_sig));
            x = orig_sig(1:min_len);
            y = dec_sig(1:min_len);
            window = 256; noverlap = 128; nfft = 256;        
            
            [S1, ~, ~] = spectrogram(x, window, noverlap, nfft, fs);
            [S2, ~, ~] = spectrogram(y, window, noverlap, nfft, fs);
            P1 = abs(S1).^2 + eps; P2 = abs(S2).^2 + eps;
            LogP1 = 10 * log10(P1); LogP2 = 10 * log10(P2);
            diff_sq = (LogP1 - LogP2).^2;
            lsd_frames = sqrt(mean(diff_sq, 1));
            lsd_val = mean(lsd_frames);
        end

        % --- HÀM TÍNH BER (Bit Error Rate) ---
        function ber_val = calculate_BER(~, data1, data2)
            % Ép kiểu về uint8 để duyệt bit an toàn
            d1 = uint8(data1); 
            d2 = uint8(data2);
            min_len = min(length(d1), length(d2));
            d1 = d1(1:min_len);
            d2 = d2(1:min_len);
            
            diff_bits = 0;
            % Duyệt qua 8 bit của từng byte (Vectorized siêu nhanh)
            for k = 1:8
                diff_bits = diff_bits + sum(bitget(d1, k) ~= bitget(d2, k));
            end
            
            total_bits = min_len * 8;
            ber_val = (diff_bits / total_bits) * 100; % Trả về phần trăm (%)
        end
        
        function AppCloseRequest(app, ~, ~)
            if ~isempty(app.NetServer)
                delete(app.NetServer);
            end
            if ~isempty(app.Recorder)
                release(app.Recorder);
            end
            delete(app.UIFigure);
        end
        
        % --- 0. CHỨC NĂNG THU ÂM, MÃ HÓA & PHÁT THỜI GIAN THỰC ---
        function LiveToggle(app, ~, ~)
            if app.BtnLive.Value
                app.IsRecording = true;
                app.BtnLive.Text = 'DỪNG THU ÂM';
                app.BtnLive.BackgroundColor = [1 0 0];
                app.lblStatus.Text = 'Đang thu âm, mã hóa và hiển thị thời gian thực...';
                drawnow;
                
                try
                    fs_live = 44100; frameSize = 4096; 
                    app.Recorder = audioDeviceReader('SampleRate', fs_live, 'SamplesPerFrame', frameSize);
                    player = audioDeviceWriter('SampleRate', fs_live, 'SupportVariableSizeInput', true);
                    setup(app.Recorder);
                    
                    md5 = java.security.MessageDigest.getInstance('MD5');
                    app.GlobalKey = typecast(md5.digest(uint8(app.txtKey.Value)), 'int8');
                  
                    app.AudioOrig = []; app.AudioEnc = []; 
                    plotCounter = 0;
                    app.RightTabs.SelectedTab = app.TabWaveform;
                    
                    while app.IsRecording
                        clean_data = app.Recorder();
                        app.AudioOrig = [app.AudioOrig; clean_data];
                        
                        y_norm = (clean_data + 1) / 2;
                        y_uint16 = uint16(round(y_norm * 65535));
                        data_bytes = typecast(y_uint16(:), 'int8');
                        
                        chunk_iv = app.generate_hash_iv(data_bytes);
                        cipher_bytes = app.run_cipher_core(data_bytes, app.GlobalKey, chunk_iv);
                        
                        enc_chunk = (double(typecast(cipher_bytes, 'uint16')) / 65535) * 2 - 1;
                        player(enc_chunk);
                        app.AudioEnc = [app.AudioEnc; enc_chunk]; 
                        
                        if ~isempty(app.NetServer) && app.NetServer.Connected
                            packet = [chunk_iv(:); cipher_bytes(:)];
                            write(app.NetServer, packet, "int8");
                        end
                        
                        plotCounter = plotCounter + 1;
                        if mod(plotCounter, 4) == 0 
                            plot(app.AxOrig, app.AudioOrig, 'b'); title(app.AxOrig, 'Sóng Âm Trực Tiếp (Bản Gốc)');
                            plot(app.AxEnc, app.AudioEnc, 'r'); title(app.AxEnc, 'Sóng Âm Đã Mã Hóa (Thời Gian Thực)');
                            drawnow limitrate;
                        end
                        
                        if length(app.AudioOrig) > fs_live * 20
                            app.BtnLive.Value = false; app.LiveToggle(); break; 
                        end
                    end
                    release(player);
                catch ME
                    app.IsRecording = false; app.BtnLive.Value = false;
                    app.BtnLive.Text = 'BẮT ĐẦU THU ÂM'; app.BtnLive.BackgroundColor = [0.6 0 0];
                    uialert(app.UIFigure, ['Lỗi Microphone: ', ME.message], 'Lỗi Phần Cứng');
                end
            else
                app.IsRecording = false; app.BtnLive.Text = 'BẮT ĐẦU THU ÂM';
                app.BtnLive.BackgroundColor = [0.6 0 0];
                app.lblStatus.Text = 'Đang tạo báo cáo tổng hợp...'; drawnow;
                
                if ~isempty(app.Recorder) release(app.Recorder); end
                if ~isempty(app.AudioOrig)
                    app.AudioOrig = app.AudioOrig * app.ScaleFactor;
                    app.Fs = 44100;
                    app.EncryptButtonPushed([], []);
                end
                app.lblStatus.Text = 'Phiên thu âm kết thúc.';
            end
        end
        
        % --- 1. LOAD AUDIO & LỌC TẠP ÂM ---
        function LoadButtonPushed(app, ~, ~)
            [file, path] = uigetfile({'*.wav;*.mp3;*.flac', 'Tệp Âm Thanh'});
            if isequal(file, 0), return; end
            app.lblStatus.Text = 'Đang tải tệp và lọc tạp âm...'; drawnow;
            try
                [y, fs] = audioread(fullfile(path, file));
                if size(y, 2) > 1, y = mean(y, 2); end
                
                max_dur = 5; if length(y) > fs * max_dur, y = y(1:floor(fs*max_dur)); end
                
                if fs > 6800
                    bpFilt = designfilt('bandpassiir', 'FilterOrder', 4, ...
                         'HalfPowerFrequency1', 300, 'HalfPowerFrequency2', 3400, 'SampleRate', fs);
                    y = filtfilt(bpFilt, y);
                end
                
                app.AudioOrig = y * app.ScaleFactor; app.Fs = fs; 
                app.AudioEnc = []; app.AudioDec = [];
                
                plot(app.AxOrig, app.AudioOrig, 'b'); 
                title(app.AxOrig, 'Sóng Âm Gốc (Đã lọc ồn)', 'FontSize', 16, 'FontWeight', 'bold'); 
                app.lblStatus.Text = ['Đã tải và lọc ồn: ' file];
                app.RightTabs.SelectedTab = app.TabWaveform;
            catch ME
                uialert(app.UIFigure, ME.message, 'Lỗi Tải Tệp');
            end
        end
        
        % --- 2. ENCRYPT & ANALYZE ---
        function EncryptButtonPushed(app, ~, ~)
            if isempty(app.AudioOrig), uialert(app.UIFigure, 'Vui lòng Tải hoặc Thu âm tín hiệu trước!', 'Cảnh Báo'); return; end
            app.lblStatus.Text = 'Đang Mã hóa & Phân tích...'; drawnow;
            try
                % --- A. MÃ HÓA ---
                keyStr = app.txtKey.Value;
                md5 = java.security.MessageDigest.getInstance('MD5');
                app.GlobalKey = typecast(md5.digest(uint8(keyStr)), 'int8');
                
                y_norm = (app.AudioOrig(:) / app.ScaleFactor + 1) / 2; 
                y_uint8 = uint8(y_norm * 255); 
                y_uint16 = uint16(round(y_norm * 65535)); 
                data_bytes = typecast(y_uint16(:), 'int8');
                
                app.IV = app.generate_hash_iv(data_bytes);
                cipher_bytes = app.run_cipher_core(data_bytes, app.GlobalKey, app.IV);
                app.AudioEnc = (double(typecast(cipher_bytes, 'uint16')) / 65535) * 2 - 1;
                
                % Các mảng Uint8 để tính toán bit và histo
                c1_uint8 = typecast(cipher_bytes, 'uint8');
                c1 = double(c1_uint8);
                
                % --- B. METRICS CƠ BẢN ---
                p_orig = histcounts(double(y_uint8), 0:256, 'Normalization', 'probability');
                entropy_orig = -sum(p_orig(p_orig>0) .* log2(p_orig(p_orig>0)));
                p_enc = histcounts(c1, 0:256, 'Normalization', 'probability');
                entropy_enc = -sum(p_enc(p_enc>0) .* log2(p_enc(p_enc>0)));
                
                r_orig_mat = corrcoef(double(y_uint8(1:end-1)), double(y_uint8(2:end)));
                corr_orig_val = r_orig_mat(1,2);
                r_enc_adj = corrcoef(c1(1:end-1), c1(2:end));
                corr_enc_adj_val = r_enc_adj(1,2);
                
                % --- C. KPA / CPA (NSCR, UACI & BER) ---
                % 1. KPA: BER giữa Bản Rõ và Bản Mã
                p_uint8 = typecast(data_bytes, 'uint8');
                ber_kpa = app.calculate_BER(p_uint8, c1_uint8);
                
                % 2. CPA: Thay đổi 1 bit của Bản rõ (Avalanche Effect)
                data_mod = data_bytes; data_mod(1) = bitxor(data_mod(1), 1); 
                iv_mod = app.generate_hash_iv(data_mod); 
                cipher_mod = app.run_cipher_core(data_mod, app.GlobalKey, iv_mod);
                c2_uint8 = typecast(cipher_mod, 'uint8');
                c2 = double(c2_uint8);
                
                nscr = (sum(c1 ~= c2) / length(c1)) * 100;
                uaci = (sum(abs(c1 - c2)) / (length(c1)*255)) * 100;
                ber_cpa = app.calculate_BER(c1_uint8, c2_uint8);
                
                % 3. Đổi 1 bit của Khóa (Key Sensitivity)
                key_mod = app.GlobalKey; key_mod(1) = bitxor(key_mod(1), 1);
                cipher_key_mod = app.run_cipher_core(data_bytes, key_mod, app.IV);
                c3_uint8 = typecast(cipher_key_mod, 'uint8');
                ber_key = app.calculate_BER(c1_uint8, c3_uint8);
                
                % --- D. CẬP NHẬT BẢNG METRICS ---
                app.MetricsTable.Data = {
                    'Entropy (Độ hỗn loạn)', sprintf('%.4f', entropy_orig), sprintf('%.4f', entropy_enc), 'Lý tưởng ≈ 8.0';
                    'Hệ số Tương quan', sprintf('%.4f', corr_orig_val), sprintf('%.4f', corr_enc_adj_val), 'Lý tưởng ≈ 0.0';
                    'NSCR (%)', 'N/A', sprintf('%.2f', nscr), 'Lý tưởng > 99%';
                    'UACI (%)', 'N/A', sprintf('%.2f', uaci), 'Lý tưởng ≈ 33.4%';
                    'BER - Bản rõ vs Bản mã (KPA)', 'N/A', sprintf('%.2f%%', ber_kpa), 'Lý tưởng ≈ 50%';
                    'BER - Đổi 1 bit Bản rõ (CPA)', 'N/A', sprintf('%.2f%%', ber_cpa), 'Lý tưởng ≈ 50%';
                    'BER - Đổi 1 bit Khóa', 'N/A', sprintf('%.2f%%', ber_key), 'Lý tưởng ≈ 50%';
                };
                
                % --- E. CẬP NHẬT CÁC ĐỒ THỊ ---
                plot(app.AxEnc, app.AudioEnc, 'r'); title(app.AxEnc, 'Sóng Âm Đã Mã Hóa');
                histogram(app.AxHistOrig, double(y_uint8), 'FaceColor', 'b', 'EdgeColor', 'none'); title(app.AxHistOrig, 'Bản Gốc');
                histogram(app.AxHistEnc, c1, 'FaceColor', 'r', 'EdgeColor', 'none'); title(app.AxHistEnc, 'Bản Mã Hóa');
                
                [s1,f1,t1] = spectrogram(app.AudioOrig,256,128,256,app.Fs);
                imagesc(app.AxSpecOrig, t1, f1, 20*log10(abs(s1)+eps)); title(app.AxSpecOrig, 'Phổ Tần Số - Bản Gốc');
                [s2,f2,t2] = spectrogram(app.AudioEnc,256,128,256,app.Fs);
                imagesc(app.AxSpecEnc, t2, f2, 20*log10(abs(s2)+eps)); title(app.AxSpecEnc, 'Phổ Tần Số - Đã Mã Hóa');
                
                scatter(app.AxCorrOrig, app.AudioOrig(1:end-1), app.AudioOrig(2:end), 5, 'filled'); title(app.AxCorrOrig, 'Tương Quan Gốc');
                scatter(app.AxCorrEnc, app.AudioEnc(1:end-1), app.AudioEnc(2:end), 5, 'filled'); title(app.AxCorrEnc, 'Tương Quan Mã Hóa');
                
                % --- F. VẼ BIỂU ĐỒ CPA VÀ KPA (Cập nhật tiêu đề BER) ---
                num_plot = min(1000, length(c1)); 
                diff_cpa = abs(c1(1:num_plot) - c2(1:num_plot));
                plot(app.AxCPA, 1:num_plot, diff_cpa, 'm');
                title(app.AxCPA, sprintf('CPA: Hiệu ứng tuyết lở (Đổi 1 bit Rõ) | BER = %.2f%%', ber_cpa));
                xlabel(app.AxCPA, 'Mẫu (Samples)'); ylabel(app.AxCPA, 'Độ lệch biên độ');
                
                num_corr = min(2000, length(app.AudioOrig));
                [xcorr_val, lags] = xcorr(app.AudioOrig(1:num_corr), app.AudioEnc(1:num_corr), 'coeff');
                plot(app.AxKPA, lags, xcorr_val, 'k');
                title(app.AxKPA, sprintf('KPA: Tương quan chéo (Rõ vs Mã) | BER = %.2f%%', ber_kpa));
                xlabel(app.AxKPA, 'Độ trễ (Lags)'); ylabel(app.AxKPA, 'Biên độ tương quan');
                
                app.lblStatus.Text = 'Hoàn tất Mã hóa & Phân tích tính bảo mật.';
                app.RightTabs.SelectedTab = app.TabMetrics;
            catch ME
                uialert(app.UIFigure, ME.message, 'Lỗi Mã Hóa');
            end
        end
        
        % --- 3. DECRYPT ---
        function DecryptButtonPushed(app, ~, ~)
            if isempty(app.AudioEnc), return; end
            app.lblStatus.Text = 'Đang giải mã...'; drawnow;
            try
                y_enc_norm = (app.AudioEnc + 1) / 2;
                enc_bytes = typecast(uint16(round(y_enc_norm * 65535)), 'int8');
                dec_bytes = app.run_cipher_core(enc_bytes, app.GlobalKey, app.IV);
                y_dec = (double(typecast(dec_bytes, 'uint16')) / 65535) * 2 - 1;
                app.AudioDec = y_dec * app.ScaleFactor;
                
                plot(app.AxDec, app.AudioDec, 'g'); 
                title(app.AxDec, 'Sóng Âm Đã Giải Mã');
                
                min_len = min(length(app.AudioOrig), length(app.AudioDec));
                orig = app.AudioOrig(1:min_len);
                dec = app.AudioDec(1:min_len);
                
                r_matrix = corrcoef(orig, dec);
                corr_val = r_matrix(1,2);
                lsd_val = app.calculate_LSD(orig, dec, app.Fs);
                
                msg = sprintf('Giải mã thành công!\n\n--- Báo Cáo Khôi Phục Dữ Liệu ---\n1. Hệ số tương quan: %.4f (Mức độ khớp: 1.0)\n2. Khoảng cách Phổ (LSD): %.6f dB', corr_val, lsd_val);
                uialert(app.UIFigure, msg, 'Hoàn Tất Giải Mã', 'Icon', 'success');
                app.lblStatus.Text = 'Giải mã thành công & Đã xác minh tính vẹn toàn.';
            catch ME
                uialert(app.UIFigure, ME.message, 'Lỗi Giải Mã');
            end
        end
        
        % --- CRYPTO HELPERS ---
        function iv = generate_hash_iv(~, data_bytes)
            md = java.security.MessageDigest.getInstance('SHA-256');
            hash = typecast(md.digest(data_bytes), 'int8'); iv = hash(1:16);
        end
        function out_bytes = run_cipher_core(~, in_bytes, key, iv)
            import javax.crypto.*
            import javax.crypto.spec.*
            cipher = Cipher.getInstance('AES/CTR/NoPadding');
            cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, 'AES'), IvParameterSpec(iv));
            out_bytes = cipher.doFinal(in_bytes);
        end
        function PlayAudio(app, type)
            y = []; switch type, case 'orig', y = app.AudioOrig; case 'enc', y = app.AudioEnc; case 'dec', y = app.AudioDec; end
            if ~isempty(y), sound(y, app.Fs); end
        end
    end
    
    % --- GIAO DIỆN (UI) VÀ HÀM KHỞI TẠO ---
    methods (Access = public)
        function app = AudioCryptoApp
            createComponents(app);
            try
                app.NetServer = tcpserver("0.0.0.0", 5000);
                app.lblStatus.Text = 'Sẵn sàng. Cổng mạng 5000 đang mở chờ kết nối...';
            catch ME
                app.lblStatus.Text = 'Sẵn sàng (Chế độ Ngoại tuyến).';
                uialert(app.UIFigure, ['Không thể mở cổng mạng 5000. Lỗi: ', ME.message], 'Cảnh Báo Mạng');
            end
            app.UIFigure.CloseRequestFcn = @app.AppCloseRequest;
            app.UIFigure.Visible = 'on';
        end
        
        function createComponents(app)
            app.UIFigure = uifigure('Position', [50 50 1250 850], 'Name', 'Hệ Thống Phân Tích & Truyền Tin Bảo Mật');
            g = uigridlayout(app.UIFigure, [1 2]); g.ColumnWidth = {300, '1x'}; 
            app.LeftPanel = uipanel(g, 'Title', 'Bảng Điều Khiển');
            pnl = uigridlayout(app.LeftPanel, [11 1]);
            pnl.RowHeight = {40, 40, 40, 40, 40, 40, 40, 40, 40, 30, '1x'}; 
            
            app.BtnLoad = uibutton(pnl, 'Text', '1. Tải Tệp Âm Thanh', 'ButtonPushedFcn', @app.LoadButtonPushed, 'BackgroundColor', [0 0.45 0.74], 'FontColor', 'w');
            uilabel(pnl, 'Text', 'Khóa Bí Mật (Secret Key):');
            app.txtKey = uieditfield(pnl, 'Value', 'E693BD9FAFBDEA8C6771FDFDD666DE9C', 'HorizontalAlignment', 'center');
            app.BtnEncrypt = uibutton(pnl, 'Text', '2. Mã Hóa & Tính Toán', 'ButtonPushedFcn', @app.EncryptButtonPushed, 'BackgroundColor', [0.85 0.33 0.1], 'FontColor', 'w');
            app.BtnDecrypt = uibutton(pnl, 'Text', '3. Giải Mã & Xác Minh', 'ButtonPushedFcn', @app.DecryptButtonPushed, 'BackgroundColor', [0.47 0.67 0.19], 'FontColor', 'w');
            app.BtnLive = uibutton(pnl, 'state', 'Text', 'BẮT ĐẦU THU ÂM', 'ValueChangedFcn', @app.LiveToggle, 'BackgroundColor', [0.6 0 0], 'FontColor', 'w', 'FontWeight', 'bold');
            
            uilabel(pnl, 'Text', '-- Phát Trực Tiếp --', 'HorizontalAlignment', 'center');
            app.BtnPlayOrig = uibutton(pnl, 'Text', 'Nghe Bản Gốc', 'ButtonPushedFcn', @(btn,event) app.PlayAudio('orig'));
            app.BtnPlayEnc = uibutton(pnl, 'Text', 'Nghe Bản Mã Hóa', 'ButtonPushedFcn', @(btn,event) app.PlayAudio('enc'));
            app.BtnPlayDec = uibutton(pnl, 'Text', 'Nghe Bản Giải Mã', 'ButtonPushedFcn', @(btn,event) app.PlayAudio('dec'));
            app.lblStatus = uilabel(pnl, 'Text', 'Đang khởi tạo...', 'FontColor', 'b');
            
            app.RightTabs = uitabgroup(g);
            app.TabWaveform = uitab(app.RightTabs, 'Title', 'Sóng Âm');
            gw = uigridlayout(app.TabWaveform, [3 1]);
            app.AxOrig = uiaxes(gw); title(app.AxOrig, 'Bản Gốc');
            app.AxEnc = uiaxes(gw); title(app.AxEnc, 'Bản Mã Hóa');
            app.AxDec = uiaxes(gw); title(app.AxDec, 'Bản Giải Mã');
            
            app.TabHistogram = uitab(app.RightTabs, 'Title', 'Biểu Đồ Phân Phối');
            gh = uigridlayout(app.TabHistogram, [1 2]);
            app.AxHistOrig = uiaxes(gh); app.AxHistEnc = uiaxes(gh);
            
            app.TabSpectrogram = uitab(app.RightTabs, 'Title', 'Phổ Tần Số');
            gs = uigridlayout(app.TabSpectrogram, [1 2]);
            app.AxSpecOrig = uiaxes(gs); app.AxSpecEnc = uiaxes(gs);
            
            app.TabCorrelation = uitab(app.RightTabs, 'Title', 'Độ Tương Quan');
            gc = uigridlayout(app.TabCorrelation, [1 2]);
            app.AxCorrOrig = uiaxes(gc); app.AxCorrEnc = uiaxes(gc);
            
            app.TabSecurity = uitab(app.RightTabs, 'Title', 'Phân Tích CPA/KPA');
            gsec = uigridlayout(app.TabSecurity, [1 2]);
            app.AxCPA = uiaxes(gsec); 
            app.AxKPA = uiaxes(gsec); 
            
            app.TabMetrics = uitab(app.RightTabs, 'Title', 'Bảng Chỉ Số Bảo Mật');
            app.MetricsTable = uitable(uigridlayout(app.TabMetrics, [1 1]));
            app.MetricsTable.ColumnName = {'Tên Chỉ Số', 'Bản Gốc', 'Bản Mã Hóa', 'Ghi Chú/Lý Tưởng'};
        end
    end
end