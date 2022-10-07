function I = ddplot(X,Y, plotting)

if nargin<3
    plotting = true;
end

Z = [X; Y];


n = size(X);k = n(2);n = n(1); m =size(Y);m = m(1);

Ix = zeros(n+m,1); Iy = zeros(n+m,1);
for i = 1:n+m
    zi = Z(i);
    for j = 1:n
        xj = X(j);
        Ix(i) = Ix(i)+norm(zi-xj);
    end

    for j = 1:m
        yj = X(j);
        Iy(i) = Iy(i)+norm(zi-yj);
    end
end
mdl = fitlm(Ix,Iy);

if plotting
    clf
    %plot(Ix,Iy,'o')
    hold on
    plot(mdl)

end



I.Ix = Ix;
I.Iy = Iy;

end