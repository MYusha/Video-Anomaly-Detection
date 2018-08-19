function [ finalvector ] = lay_it_2( fea, bin_no )
% transform the input of 13x13x256 activation maps into a final single
% feature of 12544
% or it could be 14x14x512

size_map = size(fea,1); %13
num_map = size(fea,3); %256

size_bin = (size_map+1)/2; % 7, in this setting
finalvector1 = [];
finalvector2 = [];
finalvector3 = [];
finalvector4 = [];

if bin_no==0
    disp('wrong port, please use lay_it instead of lay_it_2')
    return;
end

for i = 1:num_map
    finalvector1 = [finalvector1,reshape(fea(1:size_bin,1:size_bin,i),[1,size_bin^2])];
    finalvector2 = [finalvector2,reshape(fea(1:size_bin,size_bin:size_map,i),[1,size_bin^2])];
    finalvector3 = [finalvector3,reshape(fea(size_bin:size_map,1:size_bin,i),[1,size_bin^2])];
    finalvector4 = [finalvector4,reshape(fea(size_bin:size_map,size_bin:size_map,i),[1,size_bin^2])];
end
finalvector1 = finalvector1/norm(finalvector1);
finalvector2 = finalvector2/norm(finalvector2);
finalvector3 = finalvector3/norm(finalvector3);
finalvector4 = finalvector4/norm(finalvector4);

finalvector = [finalvector1,finalvector2,finalvector3,finalvector4];
clear finalvector1
clear finalvector2
clear finalvector3
clear finalvector4
end

