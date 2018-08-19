function [tot,time,bn_fea] = bottleneck_feature_2_o(videopath,bin_no,v,dataset_name)
tic;
obj = VideoReader(videopath);
tot = get(obj, 'NumberOfFrames');

run ../../matconvnet-1.0-beta25/matlab/vl_setupnn;
net = load('imagenet-vgg-f.mat');
net = vl_simplenn_tidy(net);

bn_fea = [];

for i = 1:tot
    this_frame = read(obj,i);
    im_ = single(this_frame);
    frame = imresize(im_, net.meta.normalization.imageSize(1:2));
    frame = frame - net.meta.normalization.averageImage;
    res = vl_simplenn(net, frame);
    layer_number = 15; %relu_5 layer selected: feel free to change the number
    featureVector = res(layer_number).x;
    % bin_no = 1; % which bin to extract feature, feel free to change in {0,1..4}
    final_vector = lay_it_2(featureVector,bin_no);
    % final_vector should be 1x(12544x4) vector
    bn_fea = [bn_fea,final_vector'];
end
if  strcmp(dataset_name,'avenue')
    save(['./features/bnfeaMat_relu_',num2str(v),'.mat'],'bn_fea');
else
    disp('double check dataset name');
end
time = toc;
disp(['appearance feature extraction time: ',num2str(time)]);
end
