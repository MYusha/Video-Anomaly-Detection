function [ bn_fea,tot ] = Get_app_features( params )
dataset_name = params.dataset_name;
v = params.vnum;
addpath('../PrepareData/Appearance_feature');
addpath(params.liblinearpath);
addpath('../Avenue_Dataset/');

bin_no = 1;
[tot,time,bn_fea] = bottleneck_feature_2_o(params.videopath,bin_no,v,dataset_name);
if strcmp(dataset_name,'avenue')
    load(['./features/bnfeaMat_relu_',num2str(v),'.mat']);
else
    disp('check dataset name');
end
bn_fea = double(bn_fea);
tot = size(bn_fea,2);
disp('tot is:');
disp(tot);

end

