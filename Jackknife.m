sigma = 5;
y = normrnd(0,sigma,100,1);
m = jackknife(@mean,y,1);
n = length(y);
bias = 0 %known bias formula
jbias = (n-1)* (mean(m)-mean(y,1)) % jackknife bias estimate
%%
sigma = 5;
y = normrnd(0,sigma,10000,1);
m = jackknife(@var,y,1);
n = length(y);
bias = -sigma^2/n %known bias formula
jbias = (n-1)* (mean(m)-var(y,1)) % jackknife bias estimate