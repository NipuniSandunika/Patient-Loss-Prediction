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

%% Bivariate part


%gender vs results

gender = x_train1(:, 3);
results = y_train1;

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(gender, results);
disp('Cross Tabulation between gender and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Define custom colors for each category (Result = 0, Result = 1)
colorResult0 =  [0, 0, 0.5]; % darkblue: RGB values for Result = 0
colorResult1 = [1, 0.6, 0.8]; % pink: RGB values for Result = 1

% Stacked bar plot with custom colors
figure;
h = bar(percentage_cross_tab, 'stacked');

% Set custom colors for each bar segment
h(1).FaceColor = colorResult0;
h(2).FaceColor = colorResult1;

xlabel('Gender');
ylabel('Results');
legend('Result = 0', 'Result = 1');
title('Gender vs Result');

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');



%proc vs results

proc = x_train1(:, 24);
results = y_train1;

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(proc, results);
disp('Cross Tabulation between Proc and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Define custom colors for each category (Result = 0, Result = 1)
colorResult0 = [0.6, 0.8, 1]; % Sky Blue: RGB values for Result = 0
colorResult1 = [0.6, 0, 0.8]; % Purple: RGB values for Result = 1

% Stacked bar plot with custom colors
figure;
h = bar(percentage_cross_tab, 'stacked');

% Set custom colors for each bar segment
h(1).FaceColor = colorResult0;
h(2).FaceColor = colorResult1;

xlabel('Proc');
ylabel('Results');
legend('Result = 0', 'Result = 1');
title('Proc vs Result');
xticklabels({'PTCA', 'CABG'});

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');





%comp vs results

comp = x_train1(:, 25);
results = y_train1;

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(comp, results);
disp('Cross Tabulation between comp and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Comp');
ylabel('Results');
legend('Result = 0', 'Result = 1');
title('Comp vs Result');
xticklabels({'NO', 'Yes'});

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');




%site vs results

site = x_train1(:, 14);
results = y_train1;

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(site, results);
disp('Cross Tabulation between comp and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Define custom colors for each category (Result = 0, Result = 1)
colorResult0 = [1, 1, 0]; % Yellow: RGB values for Result = 0
colorResult1 = [1, 0, 0]; % Red: RGB values for Result = 1

% Stacked bar plot with custom colors
figure;
h = bar(percentage_cross_tab, 'stacked');

% Set custom colors for each bar segment
h(1).FaceColor = colorResult0;
h(2).FaceColor = colorResult1;

xlabel('site');
ylabel('Results');
legend('Result = 0', 'Result = 1');
title('site vs Result');
xticklabels({'0001', '0002','0003','0004','0005'});

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');




% Anticol vs Result

Anticlot = x_train1(:,13);

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(Anticlot, y_train1);
disp('Cross Tabulation between Anticol and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Anticol');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Anticol vs Result');

label_mapping = {'0 - None', '1 -  Aspirin', '2 - Heparin', '3 - Warfarin'};
xticklabels(label_mapping);

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');





% Clotsolv vs Result

Clotsolv = x_train1(:, 19);

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(Clotsolv, y_train1);
disp('Cross Tabulation between Clotsolv and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Clotsolv');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Clotsolv vs Result');

label_mapping = {'0 - None', '1 - Streptokinase', '2 - Reteplase', '3 - Alteplase'};
xticklabels(label_mapping);

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');



% AgeCat Vs Result

AgeCat = x_train1(:,2);

% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(AgeCat, y_train1);
disp('Cross Tabulation between Age_Category and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Age Category');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Age Category vs Result');


% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');



% Ekg vs Result

ekg = x_train1(:, 16);

% Cross tabulation between ekg and Result
cross_tab = crosstab(ekg, results);
disp('Cross Tabulation between ekg and result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Ekg');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Ekg vs Result');
xticklabels({'No ST elevation', ' ST elevation'});


% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');



% cpk vs Result 

cpk= x_train1(:,17);

% Cross tabulation between cpk and Result
cross_tab = crosstab(cpk, results);
disp('Cross Tabulation between cpk and result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('cpk');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('cpk vs Result');
xticklabels({'Normal CPK', ' High CPK'});
% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center')


%  nitro vs Result

nitro= x_train1(:,12);

% Cross tabulation between nitro and Result
cross_tab = crosstab(nitro, results);
disp('Cross Tabulation between nitro and result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('nitro');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('nitro vs Result');
xticklabels({'NO', ' Yes'});
% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center')

% smoker vs Result

smoker = x_train1(:,6);

% Cross tabulation between nitro and Result
cross_tab = crosstab(nitro, results);
disp('Cross Tabulation between smoker and result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('smoker');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('smoker vs Result');
xticklabels({'NO', ' Yes'});
% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center')


% ide by Side Box plot between Time vs Reslt

% Assuming you have Time and result vectors
Time = x_train1(:, 15);
result = y_train1;

% Convert result to categorical variable
result = categorical(result);

% Create side-by-side boxplot
figure;
boxplot(Time, result, 'Labels', {'Survived', 'Dead'});
xlabel('Result Category');
ylabel('Time');
title('Boxplot of Time by Result Category');

BP =x_train1(:,5);
result= y_train1


% BP vs Result 

% Cross tabulation between ekg and Result
cross_tab = crosstab(BP, result);
disp('Cross Tabulation between ekg and result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('BP');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Bivariate Analysis: BP vs Result(Percentage)');
xticklabels({'Hypotension', ' Normal','Hypertension'});


% Obesity vs Result


% Cross tabulation between Anticol and y_train1
cross_tab = crosstab(Obesity, y_train1);
disp('Cross Tabulation between Clotsolv and Result:');
disp(cross_tab);

% Calculate percentages
total_per_category = sum(cross_tab, 2);
percentage_cross_tab = (cross_tab ./ total_per_category) * 100;

% Stacked bar plot with percentages
figure;
bar(percentage_cross_tab, 'stacked');
xlabel('Obesity');
ylabel('Percentage');
legend('Result = 0', 'Result = 1');
title('Bivariate Analysis: Obesity vs Result(Percentage)');

label_mapping = {'0 - No', '1 - Yes'};
xticklabels(label_mapping);

% Display the percentages on top of the bars

S = [percentage_cross_tab(:,2),percentage_cross_tab(:,1)];

text(repmat(1:numel(total_per_category), 1, 2),...
     percentage_cross_tab(:) + 1,...
     strcat(num2str(S(:), '%.1f'), '%'),...
     'HorizontalAlignment', 'center');