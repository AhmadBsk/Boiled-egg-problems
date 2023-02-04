
clear ; close all; clc

[data,txt] = xlsread('train.xlsx');


f = figure(1);
gscatter(data(:,2), data(:,3), data(:,4),'rgb','osd');
xlabel('weight');
ylabel('time');

lda = fitcdiscr(data(:,2:3),data(:,4));
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda)
ldaResubCM = confusionmat(data(:,4),ldaClass);


figure(2)
[x,y] = meshgrid(35:.1:70,0:.1:8);
x = x(:);
y = y(:);
j = classify([x y],data(:,2:3),data(:,4));
gscatter(x,y,j,'rgb','osd')



