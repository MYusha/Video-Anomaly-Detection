function [tot,time] = bottleneck_feature_2(videopath,bin_no)
tic;
obj = VideoReader(videopath);
tot = get(obj, 'NumberOfFrames');

run ../../matconvnet-1.0-beta25/matlab/vl_setupnn;
net = load('imagenet-vgg-f.mat');

bn_fea = [];

for i = 1:tot
    if round(i/100)==i/100
        disp('doing frame:');
        disp(i);
    end
    this_frame = read(obj,i);
    im_ = single(this_frame);
%     im_ = double(this_frame);
    frame = imresize(im_, net.meta.normalization.imageSize(1:2));
    frame = frame - net.meta.normalization.averageImage;
    res = vl_simplenn(net, frame);
    layer_number = 15; %relu_5 layer selected: feel free to change the number
    featureVector = res(layer_number).x;
%     featureVector = featureVector(:);
    % bin_no = 1; % which bin to extract feature, feel free to change in {0,1..4}
    final_vector = lay_it_2(featureVector,bin_no);
    % final_vector should be 1x(12544x4) vector
    bn_fea = [bn_fea,final_vector'];
    % bn_fea is 12544 x tot
end
save('./bnfeaMat_relu.mat','bn_fea');
time = toc;
end
