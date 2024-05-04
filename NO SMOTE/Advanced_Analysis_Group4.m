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


%% for df1 dataset

T1 = df1(:,26);
P1 = df1(:,[1:25]);


%split the dataset
rng(10)
[trainV1,valV1,testV1] = dividevec(P1',T1',0.20,0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';


tc1 = fitctree(x_train1,y_train1,'PredictorNames',{'age' 'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'off');
view(tc1);
view(tc1,'Mode','graph')

%% Advanced Analysis

%NEURAL NETWORK MODEL BUILDING

%% FFN
P = df1(:, 1:25); % Features (input data) %columnwise variable
T = df1(:, 26);   % Targets (output data)
p=transpose(P) % row wise variable
t=transpose(T)
[trainV,valV,testV] = dividevec(p,t,0.20,0.20);
minmax(p)
net=newff(minmax(p),[4,4,1],{'tansig','tansig','purelin'},'traingd');
%net=newff(minmax(p),[8,3,1],{'tansig','purelin','purelin'},'traingd');
%net=newff((minmax(p),[3,1],{'tansig','purelin'},'traingdx');
net.trainParam.show = 25;
net.trainParam.epochs = 600;
net.trainParam.mc=0.01;
net.trainParam.lr=0.003;
net1 = init(net);
[net2,tr]=train(net1,p,t,[],[],valV,testV);





%% PNN

P= x_train1'
Tc=y_train1'

% Recode 0 and 1 in y_train1' to 1 and 2
Tc_recoded = y_train1' + 1;

% Convert recoded target class indices to binary vectors
T = ind2vec(Tc_recoded);

% Create a new probabilistic neural network
net = newpnn(P, T);

% Simulate the trained network on the training data
Y = sim(net, P);
view(net)
Yc = vec2ind(Y);

% Assuming x_test1 is your test data
P2 = x_test1';

% Simulate the network on the test data
Y1 = sim(net, P2);

% Convert the output of the network to class indices
NewY1 = vec2ind(Y1);


% DECISION TREE

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
