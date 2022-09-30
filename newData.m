function NewData = newData(x, m)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

n = size(x); N = n(2); n = n(1); 
bands = floor(sqrt(N));

for i = 1:n
    Tables(i) = FrequencyTable(x(i,:),N,bands);
end

F = zeros(n,N);
for i = 1:n
    xi = x(i,:); xisort = sort(xi);
    for j = 1:N
        F(i,j) = (length(xisort(xisort<xi(j))))/N;
    end
end

F = F';
unif = randi([1 N],1,m);
data = zeros(m,n);

for i = 1:m
    ind = unif(i);
    Fi = F(ind,:);
    for j = 1:n
        fij = Fi(j);
        table = Tables(j);
        acumrelative = table.AcumRelativeFrequencies;
        a = sum(acumrelative<fij);
        if a == 0
            a = 1;
        end
        intervals = table.Intervals;
        data(i,j)=mean(intervals(a,:));
    end
end

%Output
NewData.Tables = Tables;
NewData.EmpiricalDistributions = F;
NewData.Data = data;
end