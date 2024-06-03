function S = splineFunction(i,f,W,t1,t2,v)

Wmid = (W(1,:) + W(2,:) + W(3,:) + W(4,:))./4;

N = cross((W(2,:) - W(1,:)),(W(4,:) - W(1,:)));
N2 = N.*N;
N_mag = sqrt(sum(N2));

N_cap = N./N_mag;
k_vec = f - Wmid;

direc = dot(N,k_vec) / abs(dot(N,k_vec));

n = direc.*N_cap;

vel = v.*n;
vel

S_hold(8,3,2) = 0;

S_hold(:,:,1) = coeffSolver(i,Wmid,t1,[0 0 0],vel);
S_hold(:,:,2) = coeffSolver(Wmid,f,(t2-t1),vel,[0 0 0]);

S = S_hold;

end