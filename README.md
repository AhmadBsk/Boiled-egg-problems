# Boiled-egg-problems
Boiled egg problems  (Solve with LDA, QDA, Naive Bayes Classifiers, decision tree, pruned decision tree)


`clear ; close all; clc
 
[data,txt] = xlsread('train.xlsx');
 
 
f = figure(1);
gscatter(data(:,2), data(:,3), data(:,4),'rgb','osd');
xlabel('weight');
ylabel('time');`

![image](https://user-images.githubusercontent.com/123794462/216533172-c1cae113-a541-4fa0-8a9e-2860e3b2ed9f.png)

`%Linear Discriminant Analysis (LDA)
lda = fitcdiscr(data(:,2:3),data(:,4));
ldaClass = resubPredict(lda);
 
%LDA resubstitution error
ldaResubErr = resubLoss(lda)
%ldaResubErr =0.0758
 
% confusion matrix
ldaResubCM = confusionmat(data(:,4),ldaClass);
%ldaResubCM =[141,6,0;18,79,0;0,14,243]
 
figure(2)
[x,y] = meshgrid(35:.1:70,0:.1:8);
x = x(:);
y = y(:);
j = classify([x y],data(:,2:3),data(:,4));
gscatter(x,y,j,'rgb','osd')`

![image](https://user-images.githubusercontent.com/123794462/216533268-bf14cfb1-8169-4261-9a5c-0c162ba67daa.png)

`%Quadratic Discriminant Analysis (QDA) 
qda = fitcdiscr(data(:,2:3), data(:,4),'DiscrimType','quadratic');
 
%QDA resubstitution error
qdaResubErr = resubLoss(qda)    %qdaResubErr =0.0758
 
rng(0,'twister');
cp = cvpartition(data(:,4),'KFold',10)
%K-fold cross validation partition
%cp.NumTestSets=10
%cp.TrainSize=[451,450,451,451,451,451,451,451,451,451]
%cp.TestSize=[50,51,50,50,50,50,50,50,50,50]
%cp.NumObservations=501
cvlda = crossval(lda,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda) %ldaCVErr =0.0778
 
cvqda = crossval(qda,'CVPartition',cp);
qdaCVErr = kfoldLoss(cvqda) %qdaCVErr =0.0778

%Naive Bayes Classifiers
nbGau = fitcnb(data(:,2:3), data(:,4));
 
%Naive Bayes Gaussian distribution resubstitution error
nbGauResubErr = resubLoss(nbGau)    %nbGauResubErr =0.0898
 
nbGauCV = crossval(nbGau, 'CVPartition',cp);
nbGauCVErr = kfoldLoss(nbGauCV) %nbGauCVErr =0.0918
 
figure(3)
labels = predict(nbGau, [x y]);
gscatter(x,y,labels,'rgb','osd')`


