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



%% Downsampling the dataset
%Given that the number of Dead patients are low we increase the number of
%instance of dead patients
df1_backup=df1;
majority_class=df1(df1(:,26)==0,:);
minority_class=df1(df1(:,26)==1,:);

majority_downsampled=datasample(majority_class,71*9,'Replace',false);

df2=[minority_class;majority_downsampled];

%1st Indexed data is from the dataset that was not upsampled or downsampled
%2nd indexed data is from the datset that was downsampled
%Changing the age variable to categorical
df3=df2(:,2:26);

%Now we can start fitting the KNN model
P=df3(:,1:24)';
T=df3(:,25)';

rng(1234);
[trainV1,valV1,testV1]=dividevec(P,T,0.2,0.2);

trainV.P=trainV1.P';
valV.P=valV1.P';
testV.P=testV1.P';

trainV.T=trainV1.T';
valV.T=valV1.T';
testV.T=testV1.T';

%fitting KNN with defualt parameters
k1=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','Distance','hamming');

%calculate prediction accuracy of initial tree
resub_k1=resubLoss(k1);

Y_k1= predict(k1,testV.P);

errR_k1= sum(testV.T~= Y_k1)/length(testV.T)

%optimizing KNN to find opitimal k, using cross validation.
rng(1234);
errR = [];
for i = 1:24
    k11=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',i,'Distance','hamming');
    cvknn1 = crossval(k11);
    errR(i)= kfoldLoss(cvknn1);
end
errR

% Plot of Validation KfoldLoss vs k
plot([1:24],errR)
title('KNN CV loss');
xlabel('k');
ylabel('CV Error');

% Minimum kFoldloss  value from cross validation
errR(1)=1;
minimum = min(errR)
% Index of the minimum k
I = find(errR==minimum)

%Thus we get the loss from k=14
%Best k
bknn1 = fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',14,'Distance','hamming');

%resubloss loss for best test
resub_best=resubLoss(bknn1)

%Error for optimal model
Y_k2= predict(bknn1,testV.P)
errR_k2= sum(testV.T~= Y_k2)/length(testV.T)

%Obtaining the confusion matrix
confusionmat(testV.T,Y_k2)