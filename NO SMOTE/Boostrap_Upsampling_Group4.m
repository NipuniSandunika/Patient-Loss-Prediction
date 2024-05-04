clc;clear;

M=dlmread('E:\3rd Year\2nd Semester\IS 3053 - Data Mining Techniques\Group 04\patient_loss.csv',',','A2..AE10001');

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
    count_missing_df1(i)=sum(ismissing(df(:,i)));
end

%Upsampling the dataset
%Given that the number of Dead patients are low we increase the number of
%instance of dead patients
df1_backup=df1;
majority_class=df1(df1(:,26)==0,:);
minority_class=df1(df1(:,26)==1,:);

minority_upsampled=datasample(minority_class,433,'Replace',true);
majority_downsampled=datasample(majority_class,3894,'Replace',false);

df1=[minority_upsampled;majority_downsampled];
df1=datasample(df1,4327,'Replace',false);


T1 = df1(:,26);
P1 = df1(:,[1:25]);

T2 = df1_backup(:,26);
P2 = df1_backup(:,[1:25]);

%split the dataset
rng(10)
[trainV1,valV1,testV1] = dividevec(P1',T1',0.20,0.20);
[trainV2,valV2,testV2] = dividevec(P2',T2',0.20,0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';

x_train2 = trainV2.P';
y_train2 = trainV2.T';
x_test2 = testV2.P';
y_test2 = testV2.T';


tc1 = fitctree(x_train1,y_train1,'PredictorNames',{'age' 'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'off');
view(tc1);
view(tc1,'Mode','graph')


%% Classification Tree

x = x_train1(:,[2:25]);
y = y_train1;

x2 = x_train2(:,[2:25]);
y2 = y_train2;

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

tree4 = fitctree(x2,y2,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on')
view(tree3,'Mode','graph');
B4 = predict(tree4,x_test2(:,[2:25]));


% to calculate the MSE  

x1 = x_test1(:,[2:25]);
y1 = y_test1;

L1=loss(tree1,x1,y1) 
L2=loss(tree2,x1,y1)
L3=loss(tree3,x1,y1) 
  
% to calculate the classification error by cross-validation
 [E1,SE1,NLEAF1,BESTLEVEL1]=cvloss(tree1) 
 [E2,SE2,NLEAF2,BESTLEVEL2]=cvloss(tree2)
 [E3,SE3,NLEAF3,BESTLEVEL3]=cvloss(tree3)
 [E4,SE4,NLEAF4,BESTLEVEL4]=cvloss(tree4)
