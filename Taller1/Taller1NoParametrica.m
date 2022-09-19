Temps = readtable('Temperaturas.txt');
Temps = table2array(Temps);
%%
%Punto 1
%Distribución empírica
A = zeros(35,1);
hold on
title("Empirical distributions for Canadian year to year temperatures")
xlabel("Temperature")
ylabel("Probability")
for i = 1:35
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    tq1 = t(find(t>=0));
    tq2 = t(find(t<0));
    Fq1 = F(find(t>=0));
    Fq2 = F(find(t<0));
    A(i) = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)
    plot(t,F)
end

%%
subplot(2,2,1)
hold on
for i = 1:9
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    tq1 = t(find(t>=0));
    tq2 = t(find(t<0));
    Fq1 = F(find(t>=0));
    Fq2 = F(find(t<0));
    A(i) = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)
    plot(t,F)
end
legend([{'Year 1','Year 2','Year 3', ...
    'Year 4','Year 5','Year 6','Year 7','Year 8','Year 9'}])
title("Years 1-9")
xlabel("Temperature")
ylabel("Probability")
subplot(2,2,2)
hold on
for i = 10:18
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    tq1 = t(find(t>=0));
    tq2 = t(find(t<0));
    Fq1 = F(find(t>=0));
    Fq2 = F(find(t<0));
    A(i) = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)
    plot(t,F)
end
legend([{'Year 10','Year 11','Year 12','Year 13', ...
    'Year 14','Year 15','Year 16','Year 17','Year 18'}])
title("Years 10-18")
xlabel("Temperature")
ylabel("Probability")
subplot(2,2,3)
hold on
for i = 19:27
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    tq1 = t(find(t>=0));
    tq2 = t(find(t<0));
    Fq1 = F(find(t>=0));
    Fq2 = F(find(t<0));
    A(i) = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)
    plot(t,F)
end
legend([{'Year 19','Year 20','Year 21','Year 22','Year 23', ...
    'Year 24','Year 25','Year 26','Year 27'}])
title("Years 19-27")
xlabel("Temperature")
ylabel("Probability")
subplot(2,2,4)
hold on
for i = 27:35
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    tq1 = t(find(t>=0));
    tq2 = t(find(t<0));
    Fq1 = F(find(t>=0));
    Fq2 = F(find(t<0));
    A(i) = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)
    plot(t,F)
end
legend([{'Year 28','Year 29','Year 30','Year 31','Year 32', ...
    'Year 33','Year 34','Year 35'}])
title("Years 28-35")
xlabel("Temperature")
ylabel("Probability")
%%
%Tabla de doble entrada con la media

means = mean(Temps,1);

doubletable = zeros(36,36);

doubletable(1,2:36) = means;

doubletable(2:36,1) = means;

for i = 2:36
    for j = 2:36
        valori = means(i-1);
        valorj = means(j-1);
        doubletable(i,j) = i-1;
        if valorj > valori
            doubletable(i,j) = j-1;
        end
       
        if i == j
            doubletable(i,j) = 0; 
        end

         

    end
end

tableMeans = array2table(doubletable);
a = linspace(0,35,36)

for i = 1:36
    a(i) = sprintf('%.0f',a(i));
end
%%
%Tabla de doble entrada con la estimación empírica de la media
means = A
doubletable = zeros(36,36);

doubletable(1,2:36) = means;

doubletable(2:36,1) = means;

for i = 2:36
    for j = 2:36
        valori = means(i-1);
        valorj = means(j-1);
        doubletable(i,j) = i-1;
        if valorj > valori
            doubletable(i,j) = j-1;
        end
        if i == j
            doubletable(i,j) = 0; 
        end
    end
end

array2table(doubletable)

%%
temp1 = Temps(:,1);
temp2 = Temps(:,2);
[F1,t1] = ecdf(temp1);
[F2,t2] = ecdf(temp2);

max(F1)==max(F2)
%%
%Punto 2
temp = Temps(:,1);
[F,t] = ecdf(temp);
tq1 = t(find(t>=0));
tq2 = t(find(t<0));
Fq1 = F(find(t>=0));
Fq2 = F(find(t<0));
estimator = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)

means = mean(Temps,1);
clf
hold on
plot (1:35, A)
plot (1:35,means)
title("Plug-in estimator vs Maximum Likelihood Estimator for the average temperature in Canada")
xlabel("Year")
ylabel("Estimated Temperature")

%%
%Punto 3
[minimum, minindex] = min(means);
[maximum, maxindex] = max(means);

[minF,mint] = ecdf(Temps(:,minindex));
[maxF,maxt] = ecdf(Temps(:,maxindex));
clf
hold on

title("Empirical distributions for the years with the most and the least " + ...
    "average temperatures")
xlabel("Temperature")
ylabel("Probability")
plot(maxt,maxF,'b')
plot(mint,minF,'g')

legend({'Year with maximum average temperature','Year with minimum average temperature'})
%%
hold on
ecdf(Temps(:,minindex),'Bounds','on')

ecdf(Temps(:,maxindex),'Bounds','on')
title("Confidence bands for empirical distributions")
legend({'Minimum mean temperature','Lower confidence bound', ...
    'Upper confidence bound','Maximum mean temperature','Lower confidence bound', ...
    'Upper confidence bound'})
xlabel("Temperature")
ylabel("Probability")
%% 
%Punto 4
n = 100
X = wblrnd(5, 0.8,n,1);
[F,t] = ecdf(X);
WeibulF = wblcdf(t,5,0.8);
min(abs(WeibulF-F))
clf
hold on
plot(t,WeibulF,'color','#EDB120')
plot(t,F,'color','#A2142F')
title("Real and empirical distribution functions")
legend({"Real","Empirical"})

xlabel("X")
ylabel("Probability")
%%
maxs = zeros(991,1);
for i = 1:1000
    X = wblrnd(5, 0.8,i,1);
    [F,t] = ecdf(X);
    WeibulF = wblcdf(t,5,0.8);
    maxs(i) = max(abs(WeibulF-F));
end
clf
plot(maxs,'color','#A2142F')
title("Glivenko Cantelli theorem demonstration")
xlabel("Sample size (n)")
ylabel("Maximum Error")
%%
%Exercise 7.
%Se toma una muestra de 1000 personas de muestras de n datos provenientes de
%una distribución. Se les piden ciertos datos.
N = 1000;
n = 10000;
%X = randn(N,n); %normal estándar
%X = -log(rand(N,n)); %exponencial
X = rand(N,n); %uniforme

X = wblrnd(5, 0.8,N,n);

%Mínimo de cada uno, cambia la distribución. Es un criterio.
minimun = min(X');
hist(minimun);

medianStat = sort(X,1);
medianStat = medianStat(500,:);
hist(medianStat)

[F,t] = ecdf(medianStat)

plot(t,F,'r')
title("Empirical distribution of the i-th statistic for a Weibull distribution")
xlabel("Temperature")
ylabel("Probability")

%%
n = 1000
sum = 0
for j = 1:n
    sum = sum + factorial(n)/factorial(n-j) * F^j
end

%%
%Exercise 12
%
means = mean(Temps,1);
[minimum, minindex] = min(means);
[minF,mint] = ecdf(Temps(:,minindex));
tempMin = Temps(:,minindex);

boot = bootstrp(10000,@max,tempMin);
esperanzaBootst3rap = mean(boot)
max(tempMin)
%confidence interval
CIB = [prctile(boot,2.5) prctile(boot,97.5)] 
%clf
%hist(boot);


%bias

jack = jackknife(@max,tempMin);
n = length(tempMin);
jbias = (n-1)* (mean(jack)-max(tempMin)) % jackknife bias estimate

varboot = var(boot)
%%
%Exercise 13
N = 1000;
n = 1000;
X = rand(N,n); %uniforme


%Mínimo de cada uno, cambia la distribución. Es un criterio.
minimum = min(X');
hist(minimum);
meanMins = mean(minimum);
expectedMean = 1/(n+1);

boot = bootstrp(1000,@var,minimum);
BootstrapExpected = mean(boot)
BootstrapDeviation = std(boot);
sampleVariance = var(minimum)
hist(boot)
%bootstrap confidence interval
CIB = [prctile(boot,2.5) prctile(boot,97.5)] 


jack = jackknife(@var,minimum);
n = length(minimum);
jbias = (n-1)* (mean(jack)-var(minimum)) % jackknife bias estimate



%%
%Exercise 15
%Robust mahalanobis distance

[minimum, minindex] = min(means);
[maximum, maxindex] = max(means);

minTemp = Temps(:,minindex);
maxTemp = Temps(:,maxindex);
dminTemp = mahal(minTemp,minTemp);


%%
%Exercise 16.2
N = 1000;
X = randn(N,1);

[F,t,Flo,Fup,D] = ecdf(X');
clf
hold on
plot(t,Fup);
%plot(t,F);
plot(t,normcdf(t,0,1))
plot(t,Flo);
legend({'Upper bound', 'Theoretical Normal Distribution','Lower bound'})
title('Confidence band for the Standard Normal Distribution')
xlabel('x')
ylabel('CDF (x)')
%%

%Cauchy distribution
r = trnd(1,100,1)
[F,t,Flo,Fup,D] = ecdf(r');
clf
hold on
plot(t,Fup);
%plot(t,F);
plot(t,tcdf(t,1))
plot(t,Flo);
legend({'Upper bound', 'Theoretical Cauchy Distribution','Lower bound'})
title('Confidence band for the Cauchy Distribution')
xlabel('x')
ylabel('CDF (x)')
%%
%Exercise 16.3

p = 0.75;
q = 0.6;
X = binopdf(0:1,1,p);
Y = binopdf(0:1,1,q);
[F,t] = ecdf(X);
tq1 = t(find(t>=0));
tq2 = t(find(t<0));
Fq1 = F(find(t>=0));
Fq2 = F(find(t<0));
estimator = trapz(tq1,1-Fq1) - trapz(tq2,Fq2)

%%
%Exercise 16.4

LSAT = [576 635 558 578 666 580 555 661 651 605 653 575 545 572 594]';
GPA = [3.39 3.3 2.81 3.03 3.44 3.07 3 3.43 3.36 3.13 3.12 2.74 2.76 2.88 3.96]';
X = [LSAT GPA];

plot(GPA,LSAT,'o','color','#D95319')
title('Correlation')
xlabel('GPA')
ylabel('LSAT')
%%
% correlation coefficient

p = sum((GPA-mean(GPA)).*(LSAT - mean(LSAT)))/sqrt(sum((GPA-mean(GPA)).^2)).*sqrt(sum((LSAT-mean(LSAT)).^2))
p = corr(GPA,LSAT)


%Standard error using influence function

%Standard error using jackknife

%Standard error using bootstrap
boot = bootstrp(1000,@corr,X);
BootstrapExpected = mean(boot);
BootstrapExpected = BootstrapExpected(2)

m = 100;
boots = zeros(m,1);
for i = 1:m
    boot = bootstrp(1000,@corr,X);
    BootstrapExpected = mean(boot);
    boots(i) = BootstrapExpected(2);
end

std(boots)


%%
means = mean(X);
clf
hold on
plot (1:35, A)
plot (1:35,means)
title("Plug-in estimator vs Maximum Likelihood Estimator for the average temperature in Canada")
xlabel("Year")
ylabel("Estimated Temperature")

