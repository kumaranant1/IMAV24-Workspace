function visualizer(i,f,W,S,t1,t2)

T = 0:0.01:t1;
n = size(T);

pa(n,3) = 0;

for Tind = 1:1:n
    ta = (Tind - 1) / 100;
    for u = 1:1:3
        for v = 1:1:8
            pa(Tind,u) = pa(Tind,u) +  S(v,u,1)*(ta^(8-v));
        end
    end
end

T = 0:0.01:(t2-t1);
n = size(T);

pb(n,3) = 0;

for Tind = 1:1:n
    tb = (Tind - 1) / 100;
    for u = 1:1:3
        for v = 1:1:8
            pb(Tind,u) = pb(Tind,u) + S(v,u,2)*(tb^(8-v));
        end
    end
end

Win(1,:) = W(1,:);
Win(2,:) = W(2,:);
Win(3,:) = W(3,:);
Win(4,:) = W(4,:);
Win(5,:) = W(1,:);

plot3(i(1),i(2),i(3),'x');
hold on;
plot3(f(1),f(2),f(3),'x');
hold on;
plot3(pa(:,1),pa(:,2),pa(:,3),'Color','g','LineWidth',3);
hold on;
plot3(Win(:,1),Win(:,2),Win(:,3),'Color','r','LineWidth',3);
hold on;
grid on;
hold on;
plot3(pb(:,1),pb(:,2),pb(:,3),'Color','g');
hold off;

end