function [tot,time] = bottleneck_feature_3(videopath,bin_no)
tic;
obj = VideoReader(videopath);
tot = get(obj, 'NumberOfFrames');

run ../../matconvnet-1.0-beta25/matlab/vl_setupnn;
load('ucf101-img-vgg16-split1.mat');
net = dagnn.DagNN.loadobj(net);
net.mode = 'test' ;

bn_fea = [];

for i = 1:tot
%     if round(i/100)==i/100
%         disp('doing frame:');
%         disp(i);
%     end
    this_frame = read(obj,i);
    im_ = single(this_frame);
%     im_ = double(this_frame);
    frame = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    frame = bsxfun(@minus, frame, net.meta.normalization.averageImage) ;
%     layer = 'relu5';
%     featureVector = activations(net, frame, layer);
    % find the variables containing the features of interest
    featureVar = 'x30';

    % tell the network to preserve these variables
    net.vars(net.getVarIndex(featureVar)).precious = true;

    % run the DagNN
    net.eval({'input', frame}) 

    % retrieve the features you are interested in 
    featureVector = net.vars(net.getVarIndex(featureVar)).value;
    
    
    % bin_no = 1; % which bin to extract feature, feel free to change in {0,1..4}
    final_vector = lay_it_3(featureVector,bin_no);
    % final_vector should be 1x(12544x4) vector
    bn_fea = [bn_fea,final_vector'];
    % bn_fea is 25088 x tot
end
% disp('feature extraction finished');
save('./bnfeaMat_relu.mat','bn_fea');
time = toc;
end
