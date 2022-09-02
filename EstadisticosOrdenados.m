%Se toma una muestra de 500 personas de muestras de n datos provenientes de
%una distribución. Se les piden ciertos datos.
N = 500;
n = 100;
%X = randn(N,n); %normal estándar
%X = -log(rand(N,n)); %exponencial
X = rand(N,n); %uniforme

%Primeros de cada uno, siguen siendo normal estandar porque todos vienen de
%la misma distribución
D1 = X(:,1);
hist(D1)
%%
%Mínimo de cada uno, cambia la distribución. Es un criterio.
D2 = min(X');
hist(D2)

%%
%Máximo de cada uno, cambia la distribución. 
D3 = max(X');
hist(D3)

%%
%Veamos que el estimador del estadistico ordenado es asintoticamente
%insesgado
N = 1000;
n = 3000; %Variar este valor, mientras más grande, más de acerca E a 0.
X = rand(N,n);
m = min(X');
hist(m)
E = mean(m)

%%
