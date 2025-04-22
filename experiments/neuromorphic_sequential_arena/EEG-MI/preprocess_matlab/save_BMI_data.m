%% 加载与保存指定用户的所有数据及标签
function save_BMI_data(subject, session_num, left_cut, right_cut, BMI_data_path, BMI_processed_save_path)
    cnt_subject = []; % 创建空矩阵，用来储存某个用户所有session的数据
    label_subject = []; %创建空矩阵，用来储存某个用户所有session的数据
    for session = 1:session_num
        [cnt_subject_session, label_subject_session] = load_BMI_data(subject, session, left_cut, right_cut, BMI_data_path); % 提取用户数据
        cnt_subject = cat(1, cnt_subject, cnt_subject_session); % 拼接各个session的数据
        label_subject = cat(1, label_subject, label_subject_session); % 拼接各个session的数据
    end
    
    % 设定数据与标签保存名称
    cnt_save_name = ['data_', int2str(subject)]; % cnt数据保存名称
    label_save_name = ['label_', int2str(subject)]; % label数据保存名称
    eval([cnt_save_name,'=cnt_subject',';']); % 将字符串转换为matlab可执行语句
    eval([label_save_name,'=label_subject',';']); % 将字符串转换为matlab可执行语句
    
    % 储存数据与标签
    save([BMI_processed_save_path,'\data_', int2str(subject),'.mat'],['data_', int2str(subject)]);
    save([BMI_processed_save_path,'\label_', int2str(subject),'.mat'],['label_', int2str(subject)]);
end