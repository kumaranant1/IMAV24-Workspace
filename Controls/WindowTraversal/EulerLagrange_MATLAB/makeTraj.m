close all;
clear;

i = input('Enter initial position of drone: ');
f = input('Enter final position of drone: ');
W1 = input('Enter first corner of window: ');
W2 = input('Enter second corner of window: ');
W3 = input('Enter third corner of window: ');
W4 = input('Enter fourth corner of window: ');

W = [W1;W2;W3;W4];

t1 = input('Enter time to reach window: ');
t2 = input('Enter time to reach final destination: ');

v = input('Enter flight speed of drone through the window: ');

S = splineFunction(i,f,W,t1,t2,v);

visualizer(i,f,W,S,t1,t2);