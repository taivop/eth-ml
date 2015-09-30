% In Matlab you can only call a script, which will invoke the first (main)
% function in that script (even if the script.m and the function within are
% named differently -> name them the same)

function [out1,out2] = FunctionExample(in1,in2)

    out1 = in1 + in2;
    
    out2 = in1*in2;

end

% Any other function declared after the main function can only be called by
% the main function, but not from outside.

% Scripts outside of current working directory must be added to path:
% Environment > Set Path