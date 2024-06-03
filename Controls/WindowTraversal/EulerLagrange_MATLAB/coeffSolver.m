function C = coeffSolver(i,f,T,vi,vf)

C_hold(8,3) = 0;
X = [i;f;vi;vf;[0 0 0];[0 0 0];[0 0 0];[0 0 0]];

Tmat(8,8) = 0;
Tmat(1,8) = 1;
Tmat(3,7) = 1;
Tmat(5,6) = 2;
Tmat(7,5) = 6;

for p = 1:1:8
    Tmat(2,p) = T^(8-p);
end

for p = 1:1:8
    Tmat(4,p) = (8-p)*T^(7-p);
end

for p = 1:1:8
    Tmat(6,p) = (8-p)*(7-p)*T^(6-p);
end

for p = 1:1:8
    Tmat(8,p) = (8-p)*(7-p)*(6-p)*T^(5-p);
end

for q = 1:1:3
    C_hold(:,q) = Tmat\X(:,q);
end

C = C_hold;

end