clear

filter_sigma = 13;
filter_size = 25;
params.feature = 'mot'; % features are 'mot' or 'app'
params.testnow = 0; % 1: Test and report AUC; 0: save scores
params.dataset_name = 'avenue';% Dataset name: 'avenue' or 'ucsd1' or 'ucsd2'
params.samplingtype = 0; % 0: no history sampling; 1: random sampling; 2: selective sampling
params.screen = 0;
result_ = [];
draw = 0;
bl = [];
bs = [];

for v = [1:7,9:17,19:21]

label_path = ['..//Avenue_Dataset/evaluation_code/testing_label_mask/',num2str(v),'_label.mat'];
load(label_path);
load(['./scores/',params.dataset_name,'_newrs_',num2str(v),'_',params.feature,'_',num2str(params.samplingtype),'_',num2str(params.screen),'.mat']);
tot = length(final_score);

labels = zeros(1,tot);
for i = 1:tot
    mask = cell2mat(volLabel(i));
    labels(i) = any(mask(:));
end

if draw
    figure;
    x = 1:1:tot;
    plot(x,final_score);
    title('smoothed anomaly score of video');
    xlabel('frames');
    ylabel('anomaly score');
    hold on;
    plot(x,labels,'Color','r');
end
[X,Y,~,auc] = perfcurve(labels,final_score,1);
result_ = [result_,auc];
bl = [bl,labels];
bs = [bs,final_score];
end
[Xw,Yw,~,aucw] = perfcurve(bl,bs,1);
disp(['auc is: ',num2str(aucw)]);
