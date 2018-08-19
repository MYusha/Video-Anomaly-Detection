function [ feaRaw, LocV3,fea_Vol,tot ] = Get_mot_features( params )
% this sript is for setting params
addpath('functions');
addpath(params.liblinearpath);
addpath('../Avenue_Dataset/');



params.H = 120;       % loaded video height size
params.W = 160;       % loaded video width size
params.patchWin = 10; % 3D patch spatial size 
params.tprLen = 5;    % 3D patch temporal length
params.BKH = 12;      % region number in height
params.BKW = 16;      % region number in width
params.srs = 10;       % spatial sampling rate in trainning video volume
params.trs = 2;       % temporal sampling rate in trainning video volume 
% params.PCAdim = 100;  % PCA Compression dimension


H = params.H;
W = params.W; 
tic;
disp(['MT_thr is: ',num2str(params.MT_thr)])

% create greyscale img feature vol.mat
if strcmp(params.dataset_name,'avenue')
load(['../Avenue_Dataset/testing_vol/vol', sprintf('%.2d', params.vnum),'.mat']);
else
    disp('check dataset name');
end


imgVol = im2double(vol);
tot = size(imgVol,3);
volBlur = imgVol; 
blurKer = fspecial('gaussian', [3,3], 1);
mask = conv2(ones(H,W), blurKer,'same');
for pp = 1 : size(imgVol,3)
     volBlur(:,:,pp) =  conv2(volBlur(:,:,pp), blurKer, 'same')./mask;
end
feaVol = abs(volBlur(:,:,1:(end-1)) - volBlur(:,:,2:end));
[feaRaw, LocV3,fea_Vol] = test_features(feaVol, params, tot);


time=toc;
fps_pre = tot/time;
disp('feature extraction done,fps is:')
disp(fps_pre);
end
