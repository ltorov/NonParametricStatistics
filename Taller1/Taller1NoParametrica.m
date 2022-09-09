Temps = readtable('Temperaturas.txt');
Temps = table2array(Temps);
%%
%Punto 1
%Distribución empírica
A = zeros(35,1);
hold on
title("Distribuciones empíricas para las temperaturas año a año")
xlabel("Temperatura")
ylabel("Probabilidad")
for i = 1:35
    temp = Temps(:,i);
    [F,t] = ecdf(temp);
    A(i) = trapz(t,1-F);
    plot(t,F)
end

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
%Punto 2


%%
%Punto 3
[minimum, minindex] = min(means);
[maximum, maxindex] = max(means);

[minF,mint] = ecdf(Temps(:,minindex));
[maxF,maxt] = ecdf(Temps(:,maxindex));
hold on
title("Distribución empírica para los años con mayor y menos temperatura " + ...
    "promedio")
xlabel("Temperatura")
ylabel("Probabilidad")
plot(maxt,maxF)
plot(mint,minF)
legend({'Año con mayor media','Año con menor media'})
%%

ecdf(Temps(:,minindex),'Bounds','on')

%% 
%Punto 4
n = 100
X = wblrnd(5, 0.8,n,1);
[F,t] = ecdf(X);
WeibulF = wblcdf(t,5,0.8);
min(abs(WeibulF-F))
clf
hold on
plot(t,WeibulF)
plot(t,F)
title("Función de distribución acumulada real comparada con la empírica")
xlabel("X")
ylabel("Probabilidad")
%%
maxs = zeros(991,1);
for i = 1:1000
    X = wblrnd(5, 0.8,i,1);
    [F,t] = ecdf(X);
    WeibulF = wblcdf(t,5,0.8);
    maxs(i) = max(abs(WeibulF-F));
end
clf
plot(maxs,'r')
title("Demostración teorema Glivenko Cantelli")
xlabel("Tamaño de muestra (n)")
ylabel("Error máximo")
%%

plot(x,wblcdf(t,5,0.8),'DisplayName','A=5, B=0.8')

legend('show','Location','southeast')
xlabel('x')
ylabel('cdf')


