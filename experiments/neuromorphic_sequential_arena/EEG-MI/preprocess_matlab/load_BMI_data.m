%% 加载指定用户的所有session数据
function [cnt_subject, class] = load_BMI_data(subject, session, left_cut, right_cut, data_path)

    %% 数据读取
    % 构建时变读取路径
    if subject < 10
        current_load_path = [data_path, 'sess0', int2str(session), '_subj0', int2str(subject), '_EEG_MI.mat'];
    else
        current_load_path = [data_path, 'sess0', int2str(session), '_subj', int2str(subject), '_EEG_MI.mat'];
    end

    datasets = load(current_load_path);
    % 读取训练、测试的CNT数据
    cnt_data_train = datasets.EEG_MI_train.x';
    cnt_data_test = datasets.EEG_MI_test.x';
    
    % 将数据加载成EEGLab格式
    train_EEG = pop_importdata('setname','train', ...
        'data', cnt_data_train, ...
        'dataformat', 'array', ...
        'srate', 1000, ...
        'nbchan', 116);

    % 进行平均参考
    train_EEG = pop_reref(train_EEG, []);
    % 进行FIR滤波
    train_EEG = pop_eegfilt(train_EEG, 1, 50);
    
    % 将数据加载成EEGLab格式
    test_EEG = pop_importdata('setname','test', ...
        'data', cnt_data_test, ...
        'dataformat', 'array', ...
        'srate', 1000, ...
        'nbchan', 62);
    test_EEG = pop_reref(test_EEG, []);
    test_EEG = pop_eegfilt(test_EEG, 1, 50);

    %% 裁切数据,创建标签
    % 读取数据标签
    train_class = datasets.EEG_MI_train.y_dec;
    test_class = datasets.EEG_MI_test.y_dec;
    class = [train_class, test_class]';
    
    % 读取裁切信号 cue
    train_cue = round(datasets.EEG_MI_train.t);
    test_cue = round(datasets.EEG_MI_test.t);
    
    % 数据裁切
    cnt_train = train_EEG.data; % cnt数据合并
    trial = size(train_cue, 2); % 数据集中的trial数目
    for tr = 1:trial
        I = cnt_train(:, (train_cue(tr) - left_cut) : (train_cue(tr) + right_cut));
        cnt_subject_train(tr, :, :) = I'; % 用户的cnt数据 格式为：[trial, channel, samplepoints]
    end
    
    cnt_test = test_EEG.data; % cnt数据合并
    trial = size(test_cue, 2); % 数据集中的trial数目
    for tr = 1:trial
        I = cnt_test(:, (test_cue(tr) - left_cut) : (test_cue(tr) + right_cut));
        cnt_subject_test(tr, :, :) = I'; % 用户的cnt数据 格式为：[trial, channel, samplepoints]
    end
    cnt_subject = cat(1, cnt_subject_train, cnt_subject_test);
end