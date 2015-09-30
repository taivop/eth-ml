%% INITIALIZATION

N = 10^2;

row = [1 2 3 4]; % Initialize row vector

col = [1;2;3;4]; % Initilize col vector

mat = [1 2;3 4]; % Initilize matrix

tic; % mark any command and press F1 to get info
for i = 1:N
    for j = 1:N
        A(i,j) = 1;
    end
end
t1 = toc;

tic;
B = ones(N,N); % View variables in workspace
t2 = toc;

% Print something to the command window
disp(['Clever Initialization is ' num2str(round(t1/t2)) 'x faster'])
% Strings can be concatenated [str1 str2 ...]

% Be aware of variable scope and lifetime!

% SEE ALSO
% NxN Matrix: A = ones(N,N)
% 1xN Column Vector: A = ones(N,1)
% Nx1 Row Vector: A = ones(1,N)

%% COMPUTATION
clear all   % Clear all variables
clc         % Clear command window
N = 10^2;

A = ones(N,N);
B = ones(N,N);
C = zeros(N,N);

% Compute matrix product C = A*B
tic;
for i = 1:N
    for j = 1:N
        for k = 1:N
            C(i,j) = C(i,j) + A(i,k)*B(k,j);
        end
    end
end
t1 = toc;

tic;
C = A*B;
t2 = toc;

disp(['Clever Matrix-Multiplication is ' num2str(round(t1/t2)) 'x faster'])

% SEE ALSO
% Add scalar c to all matrix elements: A + c
% Elementwise multiplication with scalar: c*A
% Apply functions elementwise: sqrt(A)
% Perform matrix-vector multiplication: A*b
% Matlab is sensitive to the shape of arrays!

%% ACCESSING ELEMENTS

clear all
clc
N = 10^2;

C = 100*ones(N,N); % How does C look like? -> check in workspace
C(10,20) = 0; % Access element that is in row 10 AND column 20
image(C)

waitforbuttonpress()

C = 100*ones(N,N);
C(10,[20,30]) = 0; % Access elements that are in (row 10) AND (column 20 OR 30)
image(C)            

waitforbuttonpress()

C = 100*ones(N,N);
C(10,:) = 0; % Access all elements in row 10
image(C)

waitforbuttonpress()

C = 100*ones(N,N);
C([10,20],[30,40]) = 0; % Access elements that are in (row 10 OR 20) AND (column 30 OR 40)
image(C)

waitforbuttonpress()

C = 100*ones(N,N);
C(10:20,30:40) = 0; % Access all elements that are within rows 10 to 20 AND columns 30 to 40
image(C)

waitforbuttonpress()

C = 100*ones(N,N);
mask = false(N,N);
mask(10,10) = true;
C(mask) = 0; % Access all elements that are true in mask (same shape as C!)
image(C)

%% FUNCTIONS

clear all
clc

[out1,out2] = FunctionExample(2,3); % If there are several outputs, use output array to catch them

sqrt5 = mysqrt(5,2); % Recursive Function

%% PLOTTING

clear all
clc

x = linspace(0,2*pi); % Initialize x to row vector with 100 evenly spaced values between 0 and pi
ysin = sin(x); % Perform elementwise sinus and store result in row vector ysin

waitforbuttonpress()

% Basic plot
plot(x,ysin)

waitforbuttonpress()

% Plot several graphs in one
ycos = cos(x);
plot(x,ysin,x,ycos)

waitforbuttonpress()

% Plot sinus with red cross markers and cosinus with connected green circles
plot(x,ysin,'rx',x,ycos,'go-')

%% MATH FUNCTIONS
% There are many, just hit F1 or search the web

%% ADVANCED TOOLBOXES
% Signal Processing, Image Processing, Machine Learning, you name it...

%% DATA IM&EXPORT
% Manually: Use Import Data button in GUI

clear all
clc

outdata = randi(10,10);

dlmwrite('datafile.dat',outdata,' '); % export outdata
type('datafile.dat') % view datafile
indata = dlmread('datafile.dat'); % import datafile

%%

















