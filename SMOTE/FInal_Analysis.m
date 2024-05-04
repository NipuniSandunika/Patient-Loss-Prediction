%Checking the Final Model

P=X_smote(:,2:25)';
T=y_smote';

rng(1234);
[trainV1,testV1]=dividevec(P,T,0.2);

trainV.P=trainV1.P';
testV.P=testV1.P';

trainV.T=trainV1.T';
testV.T=testV1.T';

%Classification Tree
TREE_FINAL = fitctree(trainV.P,trainV.T,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'},'Prune','on', 'MaxNumSplits',16)
view(TREE_FINAL,'Mode','graph');
Y_pred_tree = predict(TREE_FINAL,testV.P);
errR_TREE= sum(testV.T~= Y_pred_tree)/length(testV.T)
confusionmat(testV.T,Y_pred_tree)

%KNN
KNN_FINAL=fitcknn(trainV.P,trainV.T,'CategoricalPredictors','all','NumNeighbors',3,'Distance','hamming');
Y_pred_KNN= predict(KNN_FINAL,testV.P)
errR_KNN= sum(testV.T~= Y_pred_KNN)/length(testV.T)
confusionmat(testV.T,Y_pred_KNN)

%TreeBagging
forrest_final= TreeBagger(60, trainV.P, trainV.T);
Y_pred_forrest= predict(forrest_final,testV.P);
Y_pred_forrest=str2double(Y_pred_forrest); 
errR_forrest= sum(testV.T~= Y_pred_forrest)/length(testV.T)
confusionmat(testV.T,Y_pred_forrest)

%PNN
Pc= trainV.P';
Tc=trainV.T';
Pc_test=testV.P';
Tc_recoded = Tc + 1;
T = ind2vec(Tc_recoded);
net = newpnn(Pc, T);
Y_pred_PNN = sim(net, Pc_test);
Y_pred_PNN  = vec2ind(Y_pred_PNN);
Y_pred_PNN=(Y_pred_PNN - 1)';
errR_PNN= sum(testV.T~= Y_pred_PNN)/length(testV.T)
confusionmat(testV.T,Y_pred_PNN)

