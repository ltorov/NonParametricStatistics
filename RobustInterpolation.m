function Interpolation = RobustInterpolation(x, newN, containNan)

if nargin<3
    containNan = false
end
%x = readtable('return.txt');
%x = table2array(x);

n = size(x,1); N = size(x,2); prc = round(n*0.05); 
idx1 = randperm(n,prc); idx2 = randi([1,N],prc,1);
dirtyReturns = x; 

if containNan == false
    for i=1:prc
        dirtyReturns(idx1(i),idx2(i)) = NaN;
    end
end


[rows,col] = find(isnan(dirtyReturns));

missingRows = dirtyReturns(rows,:);
noMissRows = dirtyReturns; noMissRows(rows,:)= [];

corr(x)-corr(noMissRows)

a = newData(noMissRows',newN); data = a.Data;



New= [noMissRows; data];


k = size(missingRows,1);
m = size(New,1); M = size(New,2);

for i =1:k
    Diff = zeros(m,1);
    r = missingRows(i,:);
    idxr = find(isnan(r));
    r(idxr) = [];
    for j = 1:m
        newr = New(j,:);
        newr(idxr) =[];
        %Diff(j)=  sum(abs(newr-r));
        Diff(j)=  norm(newr-r);
        %Diff(j)=  norm(newr-r,inf);
    end
    [Min, minidx] = min(Diff);
    bestNewr = New(minidx(1),:);
    missingRows(i,idxr) = bestNewr(idxr);
end

completeRows = dirtyReturns;
completeRows(rows,:) = missingRows;

nansOg = zeros(prc,1); nans = zeros(prc,1);
for i=1:prc
    nansOg(i) = x(idx1(i),idx2(i));
    nans(i) = completeRows(idx1(i),idx2(i));
end

lin = fitlm(nans,nansOg)
plot(lin)

Interpolation.complete = completeRows;

end
