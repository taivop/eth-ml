% Creation : 8 November 2015
% Author   : dtedali
% Project  : ML_prj_3rd

function [ sig_feat ] = sig_1D_signature( mask )
    b = boundaries(255 - mask);
    max_s = size(b{1},1);
    idx_max = 1;
    for i = 1:size(b,1)
        if size(b{i},1) > max_s
            idx_max = i;
            max_s = size(b{i},1);
        end
    end
    [st, angle, x0, y0] = signature(b{idx_max});
    %  the warning about non-monotonicity is okay here  
    [sig_feat, x] = hist(angle,16);
end

