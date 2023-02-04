

clear ; close all; clc

[data,txt] = xlsread('train.xlsx');

f = figure;
gscatter(data(:,2), data(:,3), data(:,4),'rgb','osd');
xlabel('weight');
ylabel('time');


