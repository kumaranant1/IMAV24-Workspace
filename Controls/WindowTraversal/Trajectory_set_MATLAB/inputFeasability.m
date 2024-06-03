function Sdyn = inputFeasability(b,i,f,e,Abnd)

n = 1;

for B = 4:1:b
    c(1) = (2*(B+1)*(B-1) + B*(3*(B+1)*(B-1))^(0.5))/((B+1)*(B+2));
    c(2) = (2*(B+1)*(B-1) - B*(3*(B+1)*(B-1))^(0.5))/((B+1)*(B+2));
    fa(1) = abs(B*(c(1)^(1-(2/B)))*(f(2)-i(2))*(B-1-c(1)*(B+1))/(Abnd*(1+c(1))^3));
    fa(1) = abs(B*(c(2)^(1-(2/B)))*(f(2)-i(2))*(B-1-c(2)*(B+1))/(Abnd*(1+c(2))^3));
    Cup = (f(1) - i(1))*(e / (abs(i(2) - f(2)) - e))^(1/B);
    Clo = max(fa);
    
    if Clo < Cup
        n = n + 1;
    end
end

Sdyn = zeros([n 3]);

n = 1;

for B = 4:1:b
    c(1) = (2*(B+1)*(B-1) + B*(3*(B+1)*(B-1))^(0.5))/((B+1)*(B+2));
    c(2) = (2*(B+1)*(B-1) - B*(3*(B+1)*(B-1))^(0.5))/((B+1)*(B+2));
    fa(1) = abs(B*(c(1)^(1-(2/B)))*(f(2)-i(2))*(B-1-c(1)*(B+1))/(Abnd*(1+c(1))^3));
    fa(2) = abs(B*(c(2)^(1-(2/B)))*(f(2)-i(2))*(B-1-c(2)*(B+1))/(Abnd*(1+c(2))^3));
    Cup = (f(1) - i(1))*(e / (abs(i(2) - f(2)) - e))^(1/B);
    Clo = max(fa);
    
    if Clo < Cup
        Sdyn(n,1) = B;
        Sdyn(n,2) = Clo;
        Sdyn(n,3) = Cup;
        n = n + 1;
    end
end