%In this instance we conduct random forrest classification
P=X_smote(:,2:25)';
T=y_smote';
[trainV1,valV1,testV1]=dividevec(P,T,0.2,0.2);

trainV.P=trainV1.P';
valV.P=valV1.P';
testV.P=testV1.P';

trainV.T=trainV1.T';
valV.T=valV1.T';
testV.T=testV1.T';


% Create a TreeBagger object (Random Forest)
numTrees = 50;
forrest1= TreeBagger(numTrees, trainV.P, trainV.T);

%Predict Test valeus with the model
Y_Trees_1= predict(forrest1,testV.P);
Y_Trees_1=str2double(Y_Trees_1);

%Obtaining the errors
errR_k2= sum(testV.T~= Y_Trees_1)/length(testV.T)
%Obtaining the confusion matrix
confusionmat(testV.T,Y_Trees_1)

%Cross validating to find the best number of trees for the forrest
%optimizing KNN to find opitimal k, using cross validation.
errR = [];
folds=10;
num_of_trees=50:10:250;
for i = 1:length(num_of_trees)
    forrest1= TreeBagger(num_of_trees(i), trainV.P, trainV.T);
    Y_Trees_2= predict(forrest1,valV.P);
    Y_Trees_2=str2double(Y_Trees_2);
    errR(i)= sum(valV.T~= Y_Trees_2)/length(valV.T);
end

plot([50:10:250],errR)
title('Classification Error vs. No of Trees');
xlabel('No. of trees');
ylabel('Classification Error');

%THus we get the lowest error for 70 trees
%Creating a model with 70 trees
forrest1= TreeBagger(100, trainV.P, trainV.T);

%Predict Test valeus with the model
Y_Trees_2= predict(forrest1,testV.P);
Y_Trees_2=str2double(Y_Trees_2); 

%Obtaining the errors
errR_k2= sum(testV.T~= Y_Trees_2)/length(testV.T)
%Obtaining the confusion matrix
confusionmat(testV.T,Y_Trees_2)
