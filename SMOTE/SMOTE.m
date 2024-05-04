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

