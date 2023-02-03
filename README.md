author: AhmadReza Boskabadi 
date: 19 Feb 2021 
Email: ahmadreza.boskabadi@gmail.com

# Boiled-egg-problems
Boiled egg problems  (Solve with LDA, QDA, Naive Bayes Classifiers, decision tree, pruned decision tree)


```
clear ; close all; clc 
[data,txt] = xlsread('train.xlsx');
f = figure(1);
gscatter(data(:,2), data(:,3), data(:,4),'rgb','osd');
xlabel('weight');
ylabel('time');
```

![image](https://user-images.githubusercontent.com/123794462/216533172-c1cae113-a541-4fa0-8a9e-2860e3b2ed9f.png)

```
%Linear Discriminant Analysis (LDA)
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
gscatter(x,y,j,'rgb','osd')
```

![image](https://user-images.githubusercontent.com/123794462/216533268-bf14cfb1-8169-4261-9a5c-0c162ba67daa.png)

```
%Quadratic Discriminant Analysis (QDA) 
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
gscatter(x,y,labels,'rgb','osd')
```

![image](https://user-images.githubusercontent.com/123794462/216534733-f386a5e6-ff51-485c-957b-72f266cad43c.png)

```
% using a kernel density estimation
nbKD = fitcnb(data(:,2:3), data(:,4), 'DistributionNames','kernel', 'Kernel','box');
nbKDResubErr = resubLoss(nbKD)  %nbKDResubErr =0.0918
nbKDCV = crossval(nbKD, 'CVPartition',cp);
nbKDCVErr = kfoldLoss(nbKDCV)   %nbKDCVErr =0.0938
figure(4)
labels = predict(nbKD, [x y]);
gscatter(x,y,labels,'rgb','osd')
```

![image](https://user-images.githubusercontent.com/123794462/216534855-beb428ba-327b-409e-9c8e-c5ba277329ac.png)

```
%Decision Tree
t = fitctree(data(:,2:3),data(:,4),'PredictorNames',{'W' 'T' });
[grpname,node] = predict(t,[x y]);
 
figure(5)
gscatter(x,y,grpname,'grb','sod')
view(t,'Mode','graph');
```

![image](https://user-images.githubusercontent.com/123794462/216534944-af35558b-cafc-4d9d-855b-5bd3c6519eca.png)

```
dtResubErr = resubLoss(t)   %dtResubErr =0.0399
 
cvt = crossval(t,'CVPartition',cp);
dtCVErr = kfoldLoss(cvt)    %dtCVErr =0.0958
 
resubcost = resubLoss(t,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')
```

![image](https://user-images.githubusercontent.com/123794462/216535080-a528d3c6-9a99-4212-b7d3-3757a7493f58.png)

```
% "best" tree level with minimum cost plus one standard error
[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off
```

![image](https://user-images.githubusercontent.com/123794462/216535205-fe53999e-f9f2-4a01-b4ad-074018f1d631.png)

```
% pruned tree
pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')
cost(bestlevel+1)   %cost=0.1018
```
*Comparison of error rates in different methods*

![image](https://user-images.githubusercontent.com/123794462/216536208-6a9a4a48-3f94-4721-adc4-55c494bf35c2.png)

author: AhmadReza Boskabadi 
date: 19 Feb 2021 
Email: ahmadreza.boskabadi@gmail.com
