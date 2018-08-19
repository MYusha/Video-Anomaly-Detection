 % Run_Script of unmasking

clear 
% clc

% Initializing Configs 
% record frame per second
fps = []; 
auc = [];

params.feature = 'app'; % features are 'mot' or 'app'
params.testnow = 0; % 1: Test and report AUC; 0: save scores
params.dataset_name = 'avenue';% Dataset name: 'avenue' or 'ucsd1' or 'ucsd2'
params.samplingtype = 1; % 0: no history sampling; 1: random sampling; 2: selective sampling
params.savefile = 1;
params.screen = 1;

% params.np = 1; % controls sampling with/withough replacement
params.window = 10; % the sliding window have 2*w frames
params.stride = 5;
params.lambda = 0.5; % regularization term: in liblinear C = 1/lambda
params.k = 10; % removing k times
params.m = 50; % remove m features each time
params.block_size = 10;
params.filter_sigma = 13;
params.filter_size = 25;
params.MT_thr = 15;    % 3D patch selecting threshold 
params.liblinearpath = '../liblinear-2.11/matlab';

disp("check parameter setting: ");
disp(params);


% video processing
for v = [1:7,9:17,19:21]
    disp(['processing video ',num2str(v)]);
    params.vnum = v;
    if strcmp(params.dataset_name,'avenue')
        params.videopath = ['../Avenue_Dataset/testing_videos/',num2str(v,'%02d'),'.mp4'];
    else
        disp('check dataset name')
    end
    

    if strcmp(params.feature,'app')
        disp('running appearance pipeline');
        [ bn_fea,tot ] = Get_app_features( params );
        [ final_score,fps_pro] = Appearance_pipeline_rs(params,bn_fea,tot);
    elseif strcmp(params.feature,'mot')
        disp('running motion pipeline');
        [ feaRaw, LocV3,fea_Vol,tot ] = Get_mot_features( params );
        [  final_score,fps_pro ] = Motion_pipeline_feavol_rs(params,fea_Vol,tot );
    else
        disp('check feature name');
        continue;
    end

    [ cur_auc ] = Evaluate_auc( params,final_score);   
    if params.savefile
        params.savefilename = ['./scores/',params.dataset_name,'_newrs_',num2str(v),'_',params.feature,'_',num2str(params.samplingtype),'_',num2str(params.screen),'.mat'];
        save(params.savefilename,'final_score');
    end
    auc = [auc,cur_auc];
end
