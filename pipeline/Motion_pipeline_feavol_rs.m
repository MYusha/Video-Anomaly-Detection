function [  final_score,fps_pro ] = Motion_pipeline_feavol_rs( params,fea_Vol,tot )
%Initialize profile of current window first:
k = params.k;
m = params.m;
profile = zeros(1,k);
%Initialize anomaly score <- accuracy
score = zeros(1,tot);
in_time = zeros(1,tot);
block_size = params.block_size;
stride = params.stride;
window = params.window;
firsttime = 1;
poolsize = 50;

tic;
finalscore = 0.5*ones(1,tot);
first_time = 1;

for begin_frame = 1:stride:(tot-2*window)
    if mod(begin_frame,100)==1
        disp(begin_frame);
    end
    avg = zeros(1,4);
    if params.samplingtype>0
        flg = (begin_frame>=(poolsize+stride+window+1));
        if params.screen
            if flg
                flg = (mean(finalscore(begin_frame-stride:begin_frame+stride))>=mean(finalscore(window+2:begin_frame-stride)));
%                 flg = (mean(finalscore(begin_frame:begin_frame+window))>=(mean(finalscore(window+2:begin_frame))+3*std(finalscore(window+2:begin_frame))));
            end
        end
    else
        flg = false;
    end
    labels = [zeros(1,window+1+(block_size*flg)),ones(1,window)]';
    w0 = window+1+block_size*flg;
    w1= window;
    if flg
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
        if bin_no == 1
            a = 1;b=60;c=1;d=80;
        elseif bin_no == 2
            a = 1;b=60;c=81;d=160;
        elseif bin_no == 3
            a = 61;b=120;c=1;d=80;
        else
            a = 61;b=120;c=81;d=160;
        end
        features = reshape(fea_Vol(a:b,c:d,begin_frame:begin_frame+2*window),4800,2*window+1)';
        if flg    
            block_features = reshape(fea_Vol(a:b,c:d,history),4800,block_size)';
            features = [block_features;features];
        end
        
        for loops = 1:k
            features = normr(features);
            fea = sparse(features);
            % liblinear matlab included
            if flg
                model = train(labels,fea,sprintf('-s 0 -c %f -w0 %g -w1 %g -q',params.lambda,w0,w1));
            else
                model = train(labels,fea,sprintf('-s 0 -c %f -q',params.lambda));
            end
            [~, accuracy, ~] = predict(labels, fea, model,'-q');
            profile(loops) = accuracy(1)/100;
            % remove top 25 and bottom 25 weighted features:
            [~,sortIndex] = sort(model.w,'descend');
            maxidx = sortIndex(1:m/2);
            minidx = sortIndex(end-m/2+1:end);
            features(:,[maxidx,minidx]) = [];   
        end
%         mean of accuracy rates:
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
 
