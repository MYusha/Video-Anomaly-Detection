% apply gaussian filter on final_scores:

function new_score = gaussian_filter(old_score, sigma, size)
    % in This paper sigma = 13 and size = 25(1 second of video)
    x = linspace(-size / 2, size / 2, size);
    mask = exp(-x .^ 2 / (2 * sigma ^ 2));
    mask = mask / sum (mask);
    new_score = conv (old_score, mask, 'same');
end