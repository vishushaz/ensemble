Data = readtable('C:\...\Social_Network_Ads.csv');
Std_age = (Data.Age - mean(Data.Age))/std(Data.Age);
Data.Age = Std_age;
Std_Estimated_Salary = (Data.EstimatedSalary - mean(Data.EstimatedSalary))/std(Data.EstimatedSalary);
Data.EstimatedSalary = Std_Estimated_Salary;

%% KNN
%             ---------ClassificationKNN---------------
Classification_Model = fitcensemble(Data,'Purchased~Age+EstimatedSalary');
%Classification_Model.NumNeighbors = 5;
%         ----------------Test and Train sets--------------
%         -----------------CODE------------------------

cv = cvpartition(Classification_Model.NumObservations,'Holdout',0.2);
Cross_Validation_Model = crossval(Classification_Model,'cvpartition',cv);
%=========================Making Prediction for Test Sets==============

Predictions = predict(Cross_Validation_Model.Trained{1},Data(test(cv),1:end-1))
%% Analyzing the Prediction

result = confusionmat(Cross_Validation_Model.Y(test(cv)),Predictions)

%% Visualization Training Results

labels = unique(Data.Purchased);
classifier_name = 'Ensemble(Training Results)';

Age_range = min(Data.Age(training(cv)))-1:0.01:max(Data.Age(training(cv)))+1;
Estimated_salary_range = min(Data.EstimatedSalary(training(cv)))-1:0.01:max(Data.EstimatedSalary(training(cv)))+1;
[xx1,xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];

prediction_meshgrid = predict(Cross_Validation_Model.Trained{1}, XGrid);
gscatter(xx1(:), xx2(:), prediction_meshgrid, 'rgb');

hold on

training_data = Data(training(cv),:);
Y = ismember(training_data.Purchased,labels{1});

scatter(training_data.Age(Y),training_data.EstimatedSalary(Y),'o','MarkerEdgeColor','black','MarkerFaceColor','red');
scatter(training_data.Age(~Y),training_data.EstimatedSalary(~Y),'o','MarkerEdgeColor','black','MarkerFaceColor','green');

xlabel('Age');
ylabel('Estimated Salary');

title(classifier_name);
legend off, axis tight

legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');

%% Visualization Testing Results
labels = unique(Data.Purchased);
classifier_name = 'Ensemble(Training Results)';

Age_range = min(Data.Age(training(cv)))-1:0.01:max(Data.Age(training(cv)))+1;
Estimated_salary_range = min(Data.EstimatedSalary(training(cv)))-1:0.01:max(Data.EstimatedSalary(training(cv)))+1;
[xx1,xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];

prediction_meshgrid = predict(Cross_Validation_Model.Trained{1}, XGrid);

figure(2);
gscatter(xx1(:),xx2(:),prediction_meshgrid,'rgb');

hold on
testing_data = Data(test(cv),:);
Y = ismember(testing_data.Purchased,labels{1});

scatter(testing_data.Age(Y),testing_data.EstimatedSalary(Y),'o','MarkerEdgeColor','black','MarkerFaceColor','white');
scatter(testing_data.Age(~Y),testing_data.EstimatedSalary(~Y),'o','MarkerEdgeColor','black','MarkerFaceColor','black');

xlabel('Age');
ylabel('Estimated Salary');

title(classifier_name);
legend off, axis tight

legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
