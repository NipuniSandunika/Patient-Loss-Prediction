%% for df1 dataset

P1=X_smote(:,2:25)';
T1=y_smote';


%split the dataset
rng(10)
[trainV1,valV1,testV1] = dividevec(P1,T1,0.20,0.20);

x_train1 = trainV1.P';
y_train1 = trainV1.T';
x_test1 = testV1.P';
y_test1 = testV1.T';

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
Y1_pnn=(NewY1 - 1)';
errR_PNN= sum(y_test1~= Y1_pnn)/length(y_test1)

%Obtaining the confusion matrix
confusionmat(y_test1,Y1_pnn)
