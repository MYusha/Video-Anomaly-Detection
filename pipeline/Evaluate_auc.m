function [ auc ] = Evaluate_auc( params,final_score)
%   Detailed explanation goes here
tot = length(final_score);
v = params.vnum;
if strcmp(params.dataset_name,'avenue')
    label_path = ['../Avenue_Dataset/evaluation_code/testing_label_mask/',num2str(v),'_label.mat'];
    load(label_path);
    tot = length(volLabel);
    labels = zeros(1,tot);

    for i = 1:tot
        mask = cell2mat(volLabel(i));
        labels(i) = any(mask(:));
    end
%     load(['./labels/',num2str(v),'_frame_label.mat']);
end

[~,~,~,auc] = perfcurve(labels,final_score,1);
if (params.testnow)
    figure;
    x = 1:1:tot;
    plot(x,final_score);
    hold on;
    plot(x,labels,'Color','r');
end

end

