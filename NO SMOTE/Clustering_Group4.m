clc;clear;



M=dlmread('E:\3rd Year\2nd Semester\IS 3053 - Data Mining Techniques\Group 04\patient_loss.csv',',','A2..AE10001');
%reads the CSV file into a matrix M. It assumes that the data starts from cell A2 and extends to cell AE10001, and the data is separated by commas.

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
P1 = df1(:,[2:14,16:25]);


%split the dataset
rng(10)
[trainV1,valV1,testV1] = dividevec(P1',T1',0.20,0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';

%% kmeans
%% kmedoids([trainV.P; valV.P],i ,'Distance','hamming');
%selecting the best number of clusters
kValues = 2:10;
meanSilhouetteScores = zeros(length(kValues), 1);
% Compute silhouette scores for each value of k
for i = 1:length(kValues)
    k = kValues(i);
    idx = kmedoids(x_train1,k,'distance','hamming','replicates', 10);
    meanSilhouetteScores(i) = mean(silhouette(x_train1,idx,'hamming'));
end
% Plot the silhouette scores against the number of clusters
plot(kValues, meanSilhouetteScores, 'bo-');
xlabel('Number of clusters');
ylabel('Average silhouette score');
% Find the k value that maximizes the silhouette score
[bestScore, bestIndex] = max(meanSilhouetteScores);
bestK = kValues(bestIndex);
fprintf('Best number of clusters = %d, silhouette score = %.4f\n', bestK, bestScore);

[idx2, c] = kmedoids(x_train1, bestK, 'Distance', 'hamming', 'Replicates', 10);

figure(2)
[silh,h] = silhouette(x_train1,idx2,'hamming');
xlabel('Silhouette Value')
ylabel('Cluster')

%number of observations in each cluster
counts = histcounts(idx2, 1:max(idx2)+1);
counts