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

%% For df dataset

T = df(:,27);
P = df(:,[1:26, 28:29]);

%split the dataset
rng(10)
[trainV,valV,testV] = dividevec(P',T',0.20,0.20);

x_train = trainV.P';
y_train = trainV.T';
x_test = testV.P';
y_test = testV.T';

tc = fitctree(x_train, y_train, 'PredictorNames', {'age', 'agecat', 'gender', 'diabetes', 'bp', 'smoker', 'choles', 'active', 'obesity', 'angina', 'mi', 'nitro', 'anticlot', 'site', 'attphys', 'time', 'ekg', 'cpk', 'tropt', 'clotsolv', 'bleed', 'magnes', 'digi', 'betablk', 'proc', 'comp', 'los', 'cost'}, 'Prune', 'off');
view(tc, 'Mode', 'graph');


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