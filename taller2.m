x = readtable('return.txt');
x = table2array(x);

%1.a)

%%Primero tomamos los últimos 900 meses

x = x(1:900,:);
N = size(x,2); n = size(x,1);


%Pasamos el test de MannWhitney
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end

%Observamos que ninguno no pasa el test, entonces vamos a acortar la
%ventana de tiempo.

%%
x = x(1:450,:);
N = size(x,2); n = size(x,1);


%Pasamos el test de MannWhitney
MannWhitneyp = zeros(N,N); MannWhitneyh = zeros(N,N);
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        [p,h] = ranksum(Xi,Xj);
        MannWhitneyp(i,j) = p; MannWhitneyh(i,j) = h;
    end
end

I = MannWhitneyp <0.1

%%
%Para visualizar las curvas entre todos los activos y el área en común
%entre ellos
clf
at = zeros(N,N);
cont = 1;
for i = 1:N
    Xi = x(:,i);
    for j = 1:N
        Xj = x(:,j);
        mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
        pts = (mini:(maxi-mini)/100:maxi);
        [f1,x1] = ksdensity(Xi,pts); 
        [f2,x2] = ksdensity(Xj,pts); 
        dif = abs(f1-f2);
        are = abs(1-trapz(pts,dif));
        at(i,j) = are;
        subplot(N,N,cont)

        
        plot(x1,f1,'b')
        hold on
        plot(x2,f2,'r')
        title(are)
        cont = cont+1;
    end
end

%%

cont = 1;
for i = 1:N
    Xi = x(:,i);
    li = min(Xi); ls = max(Xi);
    xim = li:0.05:ls;
    Xipdf = fitdist(Xi,'Kernel');
    Xipdf = pdf(Xipdf,xim);
    for j = 1:N
        Xj = x(:,j);
        
        li = min(Xj); ls = max(Xj);
        xjm = li:0.05:ls;
        Xjpdf = fitdist(Xj,'Kernel');
        Xjpdf = pdf(Xjpdf,xjm);

        subplot(N,N,cont)
        plot(xim,Xipdf,'Color','b','LineWidth',2)
        hold on
        plot(xjm,Xjpdf,'Color','g','LineWidth',2)
        cont = cont+1;
    end
end

%%
%Ahora veamos las densidades de los activos que no seon de la misma
%distribucion segun el test

%Si dejamos alpha = 0.05 solo tenemos los activos 5 y 6.
%Con alpha = 0.1 tenemos los activos (2,6),(3,6),(5,6).


cont = 1;
Xi = x(:,2);
Xj = x(:,6);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,1)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

cont = cont+1;

Xi = x(:,3);
Xj = x(:,6);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,2)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

cont = cont+1;

Xi = x(:,5);
Xj = x(:,6);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,3)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

cont = cont+1;

%%
%Ahora veamos las densidades de los activos que no seon de la misma
%distribucion segun el test

%Como la mayoría lo pasan, escogemos los que tengan la mayor área en común.
%(2,5), (7,9), (3,10)

cont = 1;
Xi = x(:,2);
Xj = x(:,5);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,1)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

cont = cont+1;

Xi = x(:,7);
Xj = x(:,9);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,2)
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

cont = cont+1;

Xi = x(:,3);
Xj = x(:,10);
mini = min(min([Xi; Xj])); maxi = max(max([Xi; Xj]));
pts = (mini:(maxi-mini)/100:maxi);
[f1,x1] = ksdensity(Xi,pts); 
[f2,x2] = ksdensity(Xj,pts); 
dif = abs(f1-f2);
are = abs(1-trapz(pts,dif));
    
subplot(3,1,3)
   
plot(x1,f1,'b')
hold on
plot(x2,f2,'r')
title(are)

%%
%1.b Coger los dos que vengan de distinta distribución
%vamos a tomar los 1100 datos otra vez
%x será el activo 5 y y el activo 6.

x = readtable('return.txt');
x = table2array(x);

X = x(:,5); Y = x(:,6);

%1.b.1. modelo de regresion

linmod = fitlm(X,Y);
plot(linmod)
%%


TestDepthTukey(X(1:100,:),Y(1:100,:))



%%
%1.c
na = 7500; nb = 7500
xa = 6 + 2*rand(na,1); xb = 2+ 8*rand(nb,1);
Y = 0.2*randn()


%%
%1.d
heatmap(corr(x))
%%
%Del mapa de correlaciones escogemos los mas correlacionados(3,10).
X = x(:,3); Y = x(:,10);

linmod = fitlm(X,Y);
plot(linmod)


%%
%1.e.

vida = readtable('vida.txt');
vida = table2array(vida);

vidaX = vida(:,1:5);vidaY = vida(:,6:10);

ddplot(vidaX,vidaY)

%%
%1.f.

returns = readtable('return.txt');
returns = table2array(returns);

returnsX = returns(:,1:5);returnsY = returns(:,6:10);

ddplot(returnsX,returnsY)


