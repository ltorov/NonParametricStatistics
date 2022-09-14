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

array2table(doubletable)


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
means = mean(Temps,1);
[minimum, minindex] = min(means);
[minF,mint] = ecdf(Temps(:,minindex));
tempMin = Temps(:,minindex);

boot = bootstrp(10000,@max,tempMin);
esperanzaBootstrap = mean(boot)
max(tempMin)
%confidence interval
CIB = [prctile(boot,2.5) prctile(boot,97.5)] 
%clf
%hist(boot);


%bias

jack = jackknife(@max,tempMin);
n = length(tempMin);
jbias = (n-1)* (mean(jack)-max(tempMin)) % jackknife bias estimate
%%
%Exercise 13
N = 1000;
n = 1000;
X = rand(N,n); %uniforme


%Mínimo de cada uno, cambia la distribución. Es un criterio.
minimum = min(X');
hist(minimum);


boot = bootstrp(1000,@var,minimum);
BootstrapExpected = mean(boot)
BootstrapDeviation = std(boot);
sampleVariance = var(minimum)
hist(boot)
%bootstrap confidence interval
CIB = [prctile(boot,2.5) prctile(boot,97.5)] 


jack = jackknife(@min,y);
n = length(y);
jbias = (n-1)* (mean(jack)-var(minimum)) % jackknife bias estimate
