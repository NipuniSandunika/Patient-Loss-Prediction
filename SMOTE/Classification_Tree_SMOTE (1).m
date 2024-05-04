% DECISION TREE

%% Classification Tree

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

x = trainV.P;
y = trainV.T;

%Selecting the best Pruning criterion
% use validate set
tree1 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'off');
view(tree1);
view(tree1,'Mode','graph')
B1 = predict(tree1,valV.P);
errR_k1_tree= sum(valV.T~= B1)/length(valV.T)

tree2 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'PruneCriterion', 'impurity')
view(tree2,'Mode','graph');
B2 = predict(tree2,valV.P);
errR_k2_tree= sum(valV.T~= B1)/length(valV.T)


tree3 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on')
view(tree3,'Mode','graph');
B3 = predict(tree3,valV.P);
errR_k2_tree= sum(valV.T~= B3)/length(valV.T)

tree4 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on','PruneCriterion','error');
view(tree4);
view(tree4,'Mode','graph')
B4 = predict(tree4,valV.P);
errR_k1_tree= sum(valV.T~= B4)/length(valV.T)

%Setting the number of leaf nodes to a certain value
leafs = logspace(1,2,10);


N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(valV.P,valV.T,'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
figure(1)
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

%Defining the maximum number of splits

for n=1:30
    t = fitctree(valV.P,valV.T,'CrossVal','On',...
        'MaxNumSplits',n);
    err(n) = kfoldLoss(t);
end
figure(2)
plot(1:30,err);
xlabel('MaxNumSplits');
ylabel('cross-validated error');

%Checking a final Tree
tree5 = fitctree(x,y,'PredictorNames',{'agecat' 'gender' 'diabetes' 'bp' 'smoker' 'choles' 'active' 'obesity' 'angina' 'mi' 'nitro' 'anticlot' 'site' 'time' 'ekg' 'cpk' 'tropt' 'clotsolv' 'bleed' 'magnes' 'digi' 'betablk' 'proc' 'comp'}, 'Prune', 'on','PruneCriterion','error','MaxNumSplits',16);
view(tree5);
view(tree5,'Mode','graph')
B4 = predict(tree4,valV.P);
errR_k1_tree= sum(valV.T~= B4)/length(valV.T)
