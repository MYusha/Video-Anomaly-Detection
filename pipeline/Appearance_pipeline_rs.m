function [ final_score,fps_pro ] = Appearance_pipeline_rs( params,bn_fea,tot)
%Initialize profile of current window first:
k = params.k;
m = params.m;
v = params.vnum;
profile = zeros(1,k);
%Initialize anomaly score <- accuracy
score = zeros(1,tot);
in_time = zeros(1,tot);
block_size = params.block_size;
stride = params.stride;
window = params.window;
firsttime = 1;
poolsize = 50;
count = 0;

tic;
fea_dim = size(bn_fea,1)/4;
finalscore = 0.5*ones(1,tot);
for begin_frame = 1:stride:(tot-2*window)
    if mod(begin_frame,100)==1
        disp(begin_frame);
    end
    
    avg = zeros(1,4);
    if params.samplingtype>0
        flg = (begin_frame>=(poolsize+stride+window+1));
        if params.screen
            if flg
                flg = (mean(finalscore(begin_frame-stride:begin_frame-stride+window))>=mean(finalscore(window+2:begin_frame-stride)));
%                 flg = (mean(finalscore(begin_frame:begin_frame-stride+window))>=mean(finalscore(randperm(begin_frame-1,window-stride+1))))
            end
        end
    else
        flg = false;
    end
    labels = [zeros(1,window+1+(block_size*flg)),ones(1,window)]';
    w0 = window+1+block_size*flg;
    w1= window;
    if flg
        count = count+1;
        if params.samplingtype == 1
            if firsttime
                pool = sort(randperm(begin_frame-2-window,poolsize)+window+1);
                firsttime = 0;
            else
                if rand<0.3
                    pool = [pool(2:end),begin_frame-1];
                end
            end
            history = datasample(pool,block_size,'Replace',false);
        end
        
    end
    for bin_no = 1:4
        features = bn_fea((bin_no-1)*fea_dim + 1:bin_no * fea_dim,begin_frame:begin_frame+2*window)';
        if flg
             block_features = bn_fea((bin_no-1)*fea_dim + 1 : bin_no * fea_dim, history)';
             features = [block_features;features];
        end
        for loops = 1:k
            features = normr(features);
            fea = sparse(features);
            % liblinear matlab included
            if flg
%                 model = train(labels,fea,sprintf('-s 0 -c %f -q',params.lambda));
                model = train(labels,fea,sprintf('-s 0 -c %f -w0 %g -w1 %g -q',params.lambda,w0,w1));
            else
                model = train(labels,fea,sprintf('-s 0 -c %f -q',params.lambda));
            end
            [predicted_label, accuracy, ~] = predict(labels, fea, model,'-q');
            profile(loops) = accuracy(1)/100;
            % remove top 25 and bottom 25 weighted features:
            [~,sortIndex] = sort(model.w,'descend');
            maxidx = sortIndex(1:m/2);
            minidx = sortIndex(end-m/2+1:end);
            features(:,[maxidx,minidx]) = [];
            
        end
    % mean of accuracy rates:
    avg(bin_no) = mean(profile);
    end
    % assign to second half of frames:
    score(begin_frame+window+1:begin_frame+2*window) = score(begin_frame+window+1:begin_frame+2*window) + max(avg);
    in_time(begin_frame+window+1:begin_frame+2*window) = in_time(begin_frame+window+1:begin_frame+2*window) + 1;
    if begin_frame~=1
        finalscore(begin_frame-stride+window+1:begin_frame+window) = score(begin_frame-stride+window+1:begin_frame+window)./in_time(begin_frame-stride+window+1:begin_frame+window);
    end
    finalscore(begin_frame+window+1:begin_frame+2*window) = score(begin_frame+window+1:begin_frame+2*window)./in_time(begin_frame+window+1:begin_frame+2*window);
end
fps_pro = tot/toc;
final_score = gaussian_filter(finalscore, params.filter_sigma, params.filter_size);

end