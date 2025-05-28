%% Code Information
% This code is used to convert the sound wave (.wav file) into spike patterns
clear all; clc
dbstop if error

%% Load unsegmented audio

WIN_SIZE = 500;
HOP_SIZE = 500;

target_fs = 16000;  % sampling rate at 16000. if not, re-sample to 16000
Num_subband = 40;  % 40 channel?
dim_output = 360;   % Output dimension 1*36

load('Gtemp.mat');


trainfile = matfile('Training_raw_noise.mat','Writable',true); % store the encoded pattern into ''saveName1''
testfile = matfile('Testing_raw_noise.mat','Writable',true); % store the encoded pattern into ''saveName1''
minFreq = 280;
maxFreq = 8000;


kseg_train = 1;
kseg_test = 1;

%% Load unsegmented noise
[Noise_0,fs_0] = audioread('Unsegmented_Noise/00_Deg.wav');
[Noise_90,fs_90] = audioread('Unsegmented_Noise/90_Deg.wav');
[Noise_180,fs_180] = audioread('Unsegmented_Noise/180_Deg.wav');
[Noise_270,fs_270] = audioread('Unsegmented_Noise/270_Deg.wav');
Noise_0 = resample(Noise_0,target_fs,fs_0);
Noise_90 = resample(Noise_90,target_fs,fs_90);
Noise_180 = resample(Noise_180,target_fs,fs_180);
Noise_270 = resample(Noise_270,target_fs,fs_270);

SNR_noise = 0; % signal noise ratio

%% data splitting & adding noise
source_str = 'Unsegmented_Sound';
dir0 = dir(strcat(source_str,'/Seq*'));
length(dir0)
for ii = 1:length(dir0)
             filename=dir0(ii).name % take an anexample
             [x,fs] = audioread(strcat(source_str,'/',filename));
    
             s = strfind(filename,'/');
             u = strfind(filename,'_');
             d = str2double(filename(u(end)+1:u(end)+2)); %d stores the angle lable 0-355
             d = d*5;
                if d == 0
                    d =360;
                end
             split = mean(mean(abs(x)));
             z_360 = circshift(Gtemp,180+d);   
             z = resizeZ(z_360,dim_output);  %convert the label into a 1*36 gaussian curve which peaks at the angle
             len = length(x);
             NS = floor((length(x)-WIN_SIZE)/HOP_SIZE)+1; % number of samples samples can be splitted
             target_NS = 3000;  % target training/testing samples
             ratio = min(1, target_NS/NS);
                
                for ks  = 1:NS
                    SF = (ks-1)*HOP_SIZE + 1; 
                    EF = SF+WIN_SIZE-1;
                    x1 = x(SF:EF,:);
                    x1 = resample(x1,target_fs,fs);
                    m=mean(mean(abs(x1)));
                    noise_idx = rand(1);
                    if noise_idx <0.25
                        x1=addnoise(x1,Noise_0,SNR_noise);
                    elseif 0.25 <= noise_idx && noise_idx <0.5
                        x1=addnoise(x1,Noise_90,SNR_noise);
                    elseif 0.5 <= noise_idx && noise_idx <0.75
                        x1=addnoise(x1,Noise_180,SNR_noise);
                    elseif 0.75 <= noise_idx 
                        x1=addnoise(x1,Noise_270,SNR_noise);
                    end

                    if m > split
                    output_spikes=x1;
                    % store the spike pattern and the label
                    temp=rand(1);
                    if rand > 1 - ratio * 0.8
                        trainfile.X(kseg_train,1:WIN_SIZE*4) = single(output_spikes(1:WIN_SIZE*4)); %column-wise splice
                        trainfile.Y(kseg_train,1) = single(d);  % angle label
                        trainfile.Z(kseg_train,1:360)=single(z); % gaussian distributed 1*36 angle label
                        kseg_train = kseg_train + 1;
                    elseif rand < ratio * 0.2
                        testfile.X(kseg_test,1:WIN_SIZE*4) = single(output_spikes(1:WIN_SIZE*4)); %column-wise splice
                        testfile.Y(kseg_test,1) = single(d);  % angle label
                        testfile.Z(kseg_test,1:360)=single(z); % gaussian distributed 1*36 angle label
                        kseg_test = kseg_test + 1; 
                    end
                    end
                end  
end