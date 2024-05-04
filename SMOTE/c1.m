clc;clear;

M=dlmread('E:\3rd Year\2nd Semester\IS 3053 - Data Mining Techniques\Group 04\patient_loss.csv',',','A2..AE10001');

%% pre-processing part
rng(1234)
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

% Now X_smote and y_smote contain the original data along with synthetic samples generated using SMOTE

P = X_smote;
T = y_smote;

P1 = X_smote;
T1 = y_smote;

    
%split the dataset
rng(1234)
[trainV1,valV1,testV1] = dividevec(P1',T1',0.20,0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';

%% Classification Tree

x1 = x_train1(:,[2:25]);
y1 = y_train1;

x_val1 = valV1.P';
y_val1 = valV1.T';

x = x_val1(:,[2:25]);
y = y_val1;

% use validate set
tree1 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'off');
view(tree1);
view(tree1,'Mode','graph')
B1 = predict(tree1,x_test1(:,[2:25]));


tree2 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'PruneCriterion', 'impurity')
view(tree2,'Mode','graph');
B2 = predict(tree2,x_test1(:,[2:25]));


tree3 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on')
view(tree3,'Mode','graph');
B3 = predict(tree3,x_test1(:,[2:25]));


%tree4 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'CrossVal', 'on')
%view(tree4,'Mode','graph');
%B4 = predict(tree4,x_test1(:,[2:25]));


tree5 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'},'Prune','on', 'MaxNumSplits',16)
view(tree5,'Mode','graph');
B5 = predict(tree5,x_test1(:,[2:25]));


% use train set
tree1 = fitctree(x1,y1,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'off');
view(tree1);
view(tree1,'Mode','graph')
B1 = predict(tree1,x_test1(:,[2:25]));


tree2 = fitctree(x1,y1,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'PruneCriterion', 'impurity')
view(tree2,'Mode','graph');
B2 = predict(tree2,x_test1(:,[2:25]));


tree3 = fitctree(x1,y1,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on')
view(tree3,'Mode','graph');
B3 = predict(tree3,x_test1(:,[2:25]));

%tree4 = fitctree(x1,y1,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'CrossVal', 'on')
%view(tree4,'Mode','graph');
%B4 = predict(tree4,x_test1(:,[2:25]));


tree5 = fitctree(x1,y1,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'},'Prune','on', 'MaxNumSplits',16)
view(tree5,'Mode','graph');
B5 = predict(tree5,x_test1(:,[2:25]));

% to calculate the MSE  

x1 = x_test1(:,[2:25]);
y1 = y_test1;

L1=loss(tree1,x1,y1) 
L2=loss(tree2,x1,y1)
L3=loss(tree3,x1,y1) 
%L4=loss(tree4,x1,y1) 
L5=loss(tree5,x1,y1) 
  
% to calculate the classification error by cross-validation
 [E1,SE1,NLEAF1,BESTLEVEL1]=cvloss(tree1) 
 [E2,SE2,NLEAF2,BESTLEVEL2]=cvloss(tree2)
 [E3,SE3,NLEAF3,BESTLEVEL3]=cvloss(tree3)
 %[E4,SE4,NLEAF4,BESTLEVEL4]=cvloss(tree4)
 [E5,SE5,NLEAF5,BESTLEVEL5]=cvloss(tree5)

 %% drawing the barplot to get the variable that are importance

vip = predictorImportance(tree5);
figure;
bar(vip);
title("Predictor Importance Estimates");
ylabel("Estimates");
xlabel("Predictors");
xticks(1:length(tree5.PredictorNames));
xticklabels(tree5.PredictorNames);
xtickangle(45); % Adjust the angle if needed

 %% resubloss loss for best test
resub_best=resubLoss(tree5)

%Error for optimal model
Y_k2= predict(tree5,x1)
errR_k2= sum(y1~= Y_k2)/length(y1)

%Obtaining the confusion matrix
confusionmat(y1,Y_k2)
