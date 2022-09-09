n = 1000
X = 20 + 3*randn(n,1);
XOrdenados = sort(X);
Z = randn(n,1);
ZOrdenados = sort(Z);

plot(ZOrdenados,XOrdenados,'o')

%Si es normal, debe dar una línea recta

fitlm(ZOrdenados,XOrdenados) %regresión lineal
%Para ver si es normal miramos el ajuste (Rsquared)>0.98
%hipotesis nula: todos los betas son 0 (no hay modelo)
%Estadistico F
%PValue : <0.05, rechazamos, e.o.c no hay evidencia