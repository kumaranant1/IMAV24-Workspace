function createGraph(i,f,tf,S,W)

sizeS = size(S);
W_(1,:) = W(1,:);
W_(2,:) = W(2,:);
W_(3,:) = W(3,:);
W_(4,:) = W(4,:);
W_(5,:) = W(1,:);

plot3(i(1),i(2),i(3),'x');
hold on;
plot3(f(1),f(2),f(3),'X');
hold on;

t_ = 1:(tf/200):(tf + 1);
sizet = size(t_);
Traj = zeros([sizet 3]);

for n = 1:1:sizeS(1)
    for t = 1:1:sizet
        Traj(t,1) = f(1) + (i(1)-f(1))/(1 + (t/S(n,2))^S(n,1));
        Traj(t,2) = f(2) + (i(2)-f(2))/(1 + (t/S(n,4))^S(n,3));
        Traj(t,3) = f(3) + (i(3)-f(3))/(1 + (t/S(n,6))^S(n,5));
    end
    plot3(Traj(:,1),Traj(:,2),Traj(:,3),'Color','g','LineWidth',1);
    hold on
end

grid on;
plot3(W_(:,1),W_(:,2),W_(:,3),'Color','r','LineWidth',3);
hold off

end