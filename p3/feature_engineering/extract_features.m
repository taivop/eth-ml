% Creation : 7 November 2015
% Author   : dtedali
% Project  : ML_prj_3rd

clear all
close all

addpath('./PHOG')
addpath('./DIPUMToolboxV1.1.3')

%%
% Adjust this number as you change the number of features.
NUM_FEATURES = 696;

%% Generate 'train.csv'.

% Read the labels for the samples.
train_labels = csvread('train_labels.csv');

training_data = zeros(length(train_labels), NUM_FEATURES + 2);
for i = 1:length(train_labels)
    did = train_labels(i, 1);
    label = train_labels(i, 2);
    
    features = process_image('images/', did);
    
    training_data(i, :) = [did, features, label];
end

csvwrite('train.csv', training_data);

%% Generate 'test_validate.csv'.

train_ids = train_labels(:, 1);

test_validate_data = zeros(382, NUM_FEATURES + 1);

i = 1;
for did = 1:1272
    if any(train_ids == did)
        continue;
    end
  
    features = process_image('images/', did);

    test_validate_data(i, :) = [did, features];
    i = i + 1;
end

csvwrite('test_validate.csv', test_validate_data);


