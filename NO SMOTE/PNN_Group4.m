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

%% For df1 dataset

T1 = df1(:, 26);
P1 = df1(:, 1:25);

% Split the dataset
rng(1234)
[trainV1, valV1, testV1] = dividevec(P1', T1', 0.20, 0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';

%% PNN with different spread values

spread_values = [0.1,0.2,0.3,0.4, 0.5,0.8, 1.0]; % Add more spread values as needed
misclassification_errors = zeros(size(spread_values));

for spread_val_idx = 1:length(spread_values)
    spread_val = spread_values(spread_val_idx);
    disp(['Training PNN with Spread Value: ' num2str(spread_val)]);
    
    P = x_train1';
    Tc = y_train1';

    % Recode 0 and 1 in y_train1' to 1 and 2
    Tc_recoded = y_train1' + 1;

    % Convert recoded target class indices to binary vectors
    T = ind2vec(Tc_recoded);

    % Create a new probabilistic neural network with the current spread value
    net = newpnn(P, T, spread_val);

    % Simulate the trained network on the training data
    Y = sim(net, P);
    Yc = vec2ind(Y);

    % Assuming x_test1 is your test data
    P2 = x_test1';

    % Simulate the network on the test data
    Y1 = sim(net, P2);

    % Convert the output of the network to class indices
    NewY1 = vec2ind(Y1);

    % Assuming x_test1 and y_test1 are your test data
    P2 = x_test1';
    Tc_test = y_test1';

    % Recode 0 and 1 in y_test1 to 1 and 2
    Tc_test_recoded = Tc_test + 1;

    % Convert recoded target class indices to binary vectors
    T_test = ind2vec(Tc_test_recoded);

    % Simulate the trained network on the test data
    Y_test = sim(net, P2);

    % Convert the output of the network to class indices
    Yc_test = vec2ind(Y_test);

    % Compute the confusion matrix
    confMat = confusionmat(Tc_test_recoded, Yc_test);

    % Display the confusion matrix
    disp(['Confusion Matrix for Spread Value ' num2str(spread_val) ':']);
    disp(confMat);

    % Compute misclassification error
    misclassification_error = sum(confMat(:)) - sum(diag(confMat));
    misclassification_error = misclassification_error / sum(confMat(:));

    misclassification_errors(spread_val_idx) = misclassification_error;

    disp(['Misclassification Error for Spread Value ' num2str(spread_val) ': ' num2str(misclassification_error)]);
end

% Plot spread vs misclassification error
figure;
plot(spread_values, misclassification_errors, '-o');
title('Spread vs Misclassification Error');
xlabel('Spread Value');
ylabel('Misclassification Error');