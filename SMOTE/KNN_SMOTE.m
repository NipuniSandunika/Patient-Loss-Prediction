

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

%1st Indexed data is from the dataset that was not upsampled or downsampled
%2nd indexed data is from the datset that was downsampled
%Changing the age variable to categorical
%Now we can start fitting the KNN model
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

%fitting KNN with defualt parameters
k1=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','Distance','hamming');

%calculate prediction accuracy of initial tree
resub_k1=resubLoss(k1);

Y_k1= predict(k1,testV.P);

errR_k1= sum(testV.T~= Y_k1)/length(testV.T)

%optimizing KNN to find opitimal k, using cross validation.
rng(1234);
errR = [];
for i = 1:25
    k11=fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',i,'Distance','hamming');
    cvknn1 = crossval(k11);
    errR(i)= kfoldLoss(cvknn1);
end
errR

% Plot of Validation KfoldLoss vs k
plot([1:25],errR)
title('KNN CV loss');
xlabel('k');
ylabel('CV Error');

% Minimum kFoldloss  value from cross validation
errR(1)=1;
minimum = min(errR)
% Index of the minimum k
I = find(errR==minimum)

%Thus we get the loss from k=3
%Best k
bknn1 = fitcknn([trainV.P; valV.P],[trainV.T; valV.T],'CategoricalPredictors','all','NumNeighbors',3,'Distance','hamming');

%resubloss loss for best test
resub_best=resubLoss(bknn1)

%Error for optimal model
Y_k2= predict(bknn1,testV.P)
errR_k2= sum(testV.T~= Y_k2)/length(testV.T)

%Obtaining the confusion matrix
confusionmat(testV.T,Y_k2)