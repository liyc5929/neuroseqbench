function wave = addnoise(x,noise,SNR)
%% Add the noise

%% File Lengths
[sample_length,nchannel]=size(x);
noise_length=length(noise);

noise_vec=zeros(sample_length,nchannel);
%% Get Random Noise
for i = 1: nchannel
    noise_start=randi(noise_length-sample_length);
    noise_vec(:,i)=noise(noise_start:noise_start+sample_length-1);
end
%% Calculate Signal Power
s=x.^2;
%No Averaging
SPow=mean(s);

%% Calculate Noise Power
NPow=noise_vec.^2;
NPow=mean(NPow);

%% Calculate SNR Ratio
Ratio = sqrt(SPow / (10.0^(SNR / 10.0) * NPow));

%% Add the regions
wave = x + noise_vec*Ratio;
end
