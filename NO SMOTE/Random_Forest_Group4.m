clc;clear;

M=dlmread('E:\3rd Year\2nd Semester\IS 3053 - Data Mining Techniques\Group 04\patient_loss.csv',',','A2..AE10001');

%% pre-processing part

%Removing categories lesser than 0 from the dataset
Y=M(:,29);
mask=Y>0;
df=M(mask,:);

%Checking the number of unique values
unique(df(:,29))

counts=zeros(31,1);
for i=1:31
    counts(i)=sum(df(:,i)<0);
end

count_sum=sum(df(:,26)==1);
%Given that there are no values below 0 we can assume there are no
%categoris under 0 also

%checking for missing values in the new dataset
%count_missing=zeros(31,1);

%Changing result vector to a binary classification columns
one_values=sum(df(:,29)==4);

%Thus there should be 71 1 values in the result vector

for i=1:height(df)
    if df(i,29)==4;
        df(i,29)=1;
    else
        df(i,29)=0;
    end
end

%Creating backup and duplicate dataframes for df
df1=df;
df_backup=df;

sum(df(:,29))
sum(df1(:,29))
sum(df_backup(:,29))

%Remove doa and der as they only contain 0 values
idx=[17 26];
df(:,idx)=[];
%We also remove columns that are not found in the description of the
%dataset 

%We also remove variable not found in the datasheet
idx2=[15 17 26 30 31];
df1(:,idx2)=[];

%Here df1 contains ONLY columns found in the datasheet
%df contains includes all columns except der and doa

%checking for missing values in both df and df1
count_missing_df=zeros(width(df),1);
count_missing_df1=zeros(width(df1),1);

for i=1:width(df)
    count_missing_df(i)=sum(ismissing(df(:,i)));
end


for i=1:width(df1)
    count_missing_df1(i)=sum(ismissing(df1(:,i)));
end

X=df1(:,1:25);
y=df1(:,26);

% Find minority class samples
minority_indices = find(y == 1);
minority_samples = X(minority_indices, :);

% Calculate the number of minority class samples
num_minority_samples = length(minority_indices);

% Set the number of synthetic samples to generate (adjust this parameter as needed)
num_synthetic_samples = 1000;

% Set the number of nearest neighbors to consider
k_neighbors = 5;

% Initialize a matrix to store synthetic samples
synthetic_samples = zeros(num_synthetic_samples, size(X, 2));

% Perform SMOTE
for i = 1:num_synthetic_samples
    % Randomly select a minority sample
    random_index = randsample(num_minority_samples, 1);
    minority_sample = minority_samples(random_index, :);
    
    % Find k nearest neighbors of the minority sample
    distances = pdist2(minority_sample, X);
    [~, sorted_indices] = sort(distances);
    nearest_neighbors_indices = sorted_indices(2:k_neighbors+1); % Exclude itself
    
    % Randomly select one of the nearest neighbors
    nearest_neighbor_index = randsample(nearest_neighbors_indices, 1);
    nearest_neighbor = X(nearest_neighbor_index, :);
    
    % Generate synthetic sample
    synthetic_sample = minority_sample + rand(1, size(X, 2)) .* (nearest_neighbor - minority_sample);
    
    % Add synthetic sample to the matrix
    synthetic_samples(i, :) = synthetic_sample;
end

% Concatenate synthetic samples with original data
X_smote = [X; synthetic_samples];
y_smote = [y; ones(num_synthetic_samples, 1)]; % Assuming the label for synthetic samples is 1



%% Random Forest

%In this instance we conduct random forrest classification
P=X_smote(:,2:25)';
T=y_smote';

rng(1234);
[trainV1,valV1,testV1]=dividevec(P,T,0.2,0.2);

trainV.P=trainV1.P';
valV.P=valV1.P';
testV.P=testV1.P';

trainV.T=trainV1.T';
valV.T=valV1.T';
testV.T=testV1.T';


% Create a TreeBagger object (Random Forest)
numTrees = 50;
forrest1= TreeBagger(numTrees, trainV.P, trainV.T);

%Predict Test valeus with the model
Y_Trees_1= predict(forrest1,testV.P);
Y_Trees_1=str2double(Y_Trees_1);

%Obtaining the errors
errR_k2= sum(testV.T~= Y_Trees_1)/length(testV.T)
%Obtaining the confusion matrix
confusionmat(testV.T,Y_Trees_1)

%Cross validating to find the best number of trees for the forrest
%optimizing KNN to find opitimal k, using cross validation.

errR = [];
folds=10;
num_of_trees=50:10:250;
for i = 1:length(num_of_trees)
    forrest1= TreeBagger(num_of_trees(i), trainV.P, trainV.T);
    Y_Trees_2= predict(forrest1,valV.P);
    Y_Trees_2=str2double(Y_Trees_2);
    errR(i)= sum(valV.T~= Y_Trees_2)/length(valV.T);
end

plot([50:10:250],errR)
title('MSE vs. No of Trees');
xlabel('No. of trees');
ylabel('MSE');

%THus we get the lowest error for 70 trees
%Creating a model with 70 trees
forrest1= TreeBagger(70, trainV.P, trainV.T);

%Predict Test valeus with the model
Y_Trees_2= predict(forrest1,testV.P);
Y_Trees_2=str2double(Y_Trees_2);

%Obtaining the errors
errR_k2= sum(testV.T~= Y_Trees_2)/length(testV.T)
%Obtaining the confusion matrix
confusionmat(testV.T,Y_Trees_2)