clc
clear
close

%% 读取、保存 RawData 数据经过重参考、带通滤波，并最终拼接为 [samples, channels, (points = right_cut - left_cut)]
BMI_data_path = 'set to your path'; % 原始数据集读取路径
BMI_processed_save_path = 'set to your path'; %加工数据集储存路径
subject_num = 54; % 数据集用户数
session_num = 2; % 数据集session数目
left_cut = 0; % 左裁切大小：设为负
right_cut = 350 - 1; % 右裁切大小：设为正

for subject = 1:subject_num
    % 加载，保存某个用户的数据
    save_BMI_data(subject, session_num, left_cut, right_cut, BMI_data_path, BMI_processed_save_path);
end