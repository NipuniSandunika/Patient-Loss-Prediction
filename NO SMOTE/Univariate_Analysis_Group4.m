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

%% Descriptive analysis

%% Univariate analysis

%proc

proc = x_train1(:, 24);
figure;
categoryCounts = histcounts(proc);
percentages = (categoryCounts / numel(proc)) * 100;
pie(percentages)
title('Pie Chart for proc Variable')
textlabel = strcat(num2str(percentages.'), '%');
legend('PTCA', 'CABG') 

%comp

comp = x_train1(:, 25);
figure;
categoryCounts = histcounts(comp);
percentages = (categoryCounts / numel(comp)) * 100;
pie(percentages)
title('Pie Chart for comp Variable')
textlabel = strcat(num2str(percentages.'), '%');
legend('No', 'Yes') 



%site

site = x_train1(:, 14);
figure;
categoryCounts = histcounts(site);
percentages = (categoryCounts / numel(site)) * 100;
pie(percentages)
title('Pie Chart for site Variable')
textlabel = strcat(num2str(percentages.'), '%');
legend('0001', '0002','0003','0004','0005') 



%Anticlot

Anticlot = x_train1(:,13);

% Frequency table for Clotsolv
Anticlot_freq = tabulate(Anticlot);
disp('Frequency Table for Anticlot:');
disp(Anticlot_freq);

label_mapping = {'0 - None', '1 -  Aspirin', '2 - Heparin', '3 - Warfarin'};

% Bar plot for Clotsolv
figure;
bar(Anticlot_freq(:, 1), Anticlot_freq(:, 3), 'red')
xlabel('Anticlot Levels');
ylabel('Percentage');
title('Univariate Analysis: Anticlot');
xticklabels(label_mapping);



% Pie chart for Clotsolv with different colors
figure;
h = pie(Anticlot_freq(:, 3));
title('Univariate Analysis: Anticlot - Pie Chart');

% Set different colors for each level
color_palette = jet(numel(Anticlot_freq(:, 1)));
for i = 1:numel(h)/2
    h(i*2-1).FaceColor = color_palette(i, :);
end

% Create a legend
legend(cellstr(num2str(Anticlot_freq(:, 1))), 'Location', 'BestOutside');

% Display separated label descriptions below the pie chart
text(0.5, -1.2, strjoin(label_mapping, ' | '), 'HorizontalAlignment', 'center', 'FontSize', 10);



%Clotsolv
Clotsolv = x_train1(:, 19);

% Frequency table for Clotsolv
Clotsolv_freq = tabulate(Clotsolv);
disp('Frequency Table for Clotsolv:');
disp(Clotsolv_freq);

label_mapping = {'0 - None', '1 - Streptokinase', '2 - Reteplase', '3 - Alteplase'};

% Bar plot for Clotsolv
figure;
bar(Clotsolv_freq(:, 1), Clotsolv_freq(:, 3), 'red')
xlabel('Clotsolv Levels');
ylabel('Percentage');
title('Univariate Analysis: Clotsolv');
xticklabels(label_mapping);



% Pie chart for Clotsolv with different colors
figure;
h = pie(Clotsolv_freq(:, 3));
title('Univariate Analysis: Clotsolv - Pie Chart');

% Set different colors for each level
color_palette = jet(numel(Clotsolv_freq(:, 1)));
for i = 1:numel(h)/2
    h(i*2-1).FaceColor = color_palette(i, :);
end

% Create a legend
legend(cellstr(num2str(Clotsolv_freq(:, 1))), 'Location', 'BestOutside');

% Display separated label descriptions below the pie chart
text(0.5, -1.2, strjoin(label_mapping, ' | '), 'HorizontalAlignment', 'center', 'FontSize', 10);



%Age

% Example data (replace this with your actual data)
load('data.mat'); % Assuming 'p' is loaded from a .mat file
% Extract the first variable from 'p'
Age = x_train1(:,1)
% Plot the histogram
histogram(Age);
xlabel('Value'); % Label for x-axis
ylabel('Age'); % Label for y-axis
title('Histogram of Age'); % Title for the plot
% Calculate the five-number summary
min_age = min(Age);
Q1_age = quantile(Age, 0.25);
median_age = median(Age);
Q3_age = quantile(Age, 0.75);
max_age = max(Age);
% Create a table for the five-number summary
summary_table = table(min_age, Q1_age, median_age, Q3_age, max_age, ...
    'VariableNames', {'Minimum', 'Q1', 'Median', 'Q3', 'Maximum'});
% Display the table
disp(summary_table);


%AgeCat

AgeCat = x_train1(:,2);
% Plot the histogram
figure;
histogram(AgeCat);
xlabel('Value'); % Label for x-axis
ylabel('Age_CAtegory'); % Label for y-axis
title('Histogram of Age'); % Title for the plot


% Time

% Example data (replace this with your actual data)
load('data.mat'); % Assuming 'p' is loaded from a .mat file
% Extract the first variable from 'p'
Time = x_train1(:,15)
% Plot the histogram
histogram(Time);
xlabel('Value'); % Label for x-axis
ylabel('Time'); % Label for y-axis
title('Histogram of Time'); % Title for the plot
% Calculate the five-number summary for Time
min_time = min(Time);
Q1_time = quantile(Time, 0.25);
median_time = median(Time);
Q3_time = quantile(Time, 0.75);
max_time = max(Time);
% Create a table for the five-number summary
summary_table = table(min_time, Q1_time, median_time, Q3_time, max_time, ...
    'VariableNames', {'Minimum', 'Q1', 'Median', 'Q3', 'Maximum'});
% Display the table
disp(summary_table);


Time =x_train1(:,15);
ekg = x_train1(:,16);
cpk= x_train1(:,17);
tropt = x_train1(:,18);
result= y_train1
nitro= x_train1(:,12);

% ekg 

% Assuming you have ekg categorical variable
ekg = x_train1(:, 16);
% Count occurrences of each category
category_counts = histcounts(ekg);
% Get unique categories
unique_categories = unique(ekg);
% Create bar chart
figure;
bar(unique_categories, category_counts);
xlabel('Categories');
ylabel('Count');
title('Bar Chart of EKG Categories');
% Set custom labels for the x-axis ticks
xticklabels({'No ST elevation', 'ST elevation'});
%# Create pie chart
figure;
pie(category_counts, unique_categories);
title('Pie Chart of EKG Categories');
legend('No ST elevation', 'ST elevation');

% cpk
cpk= x_train1(:,17);

% Count occurrences of each category
category_counts = histcounts(cpk);

% Get unique categories
unique_categories = unique(cpk);

% Create bar chart
figure;
bar(unique_categories, category_counts);
xlabel('Categories');
ylabel('Count');
title('Bar Chart of cpk Categories');
% Set custom labels for the x-axis ticks
xticklabels({'Normal CPK', 'High CPK'});

%# Create pie chart
figure;
pie(category_counts, unique_categories);
title('Pie Chart of CPK Categories');
legend('Normal CPK', 'High CPK');


%Obesity

Obesity = x_train1(:,9);

% Frequency table for Clotsolv
Obesity_freq = tabulate(Obesity);
disp('Frequency Table for Obesity:');
disp(Obesity_freq);

label_mapping = {'0 - No', '1 -  Yes'};

% Bar plot for Obesity
figure;
bar(Obesity_freq(:, 1), Obesity_freq(:, 2), 'red')
xlabel('Obesity Levels');
ylabel('Percentage');
title('Univariate Analysis: Obesity');
xticklabels(label_mapping);



% Pie chart for Clotsolv with different colors
figure;
h = pie(Obesity_freq(:, 3));
title('Univariate Analysis: Obesity');

% Set different colors for each level
color_palette = jet(numel(Obesity_freq(:, 1)));
for i = 1:numel(h)/2
    h(i*2-1).FaceColor = color_palette(i, :);
end

% Create a legend
legend(cellstr(num2str(Obesity_freq(:, 1))), 'Location', 'BestOutside');

% Display separated label descriptions below the pie chart
text(0.5, -1.2, strjoin(label_mapping, ' | '), 'HorizontalAlignment', 'center', 'FontSize', 10);

