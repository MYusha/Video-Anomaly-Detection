function bin_number = get_bin_no( a,b )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% a and b are cube region index: a in [1,12], b in [1,16]
bin_number = 0;
if a<=6 && b<=8
    bin_number = 1;
end
if a>6 && b<=8
    bin_number = 3;
end
if a>6 && b>8
    bin_number = 4;
end
if a<=6 && b>8
    bin_number = 2;
end

end

