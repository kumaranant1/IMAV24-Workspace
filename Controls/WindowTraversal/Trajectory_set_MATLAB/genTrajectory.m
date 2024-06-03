close all;
clear

i = input('Enter initial position: ');
f = input('Enter final position: ');
e = input('Enter tolerance for final position: ');
tf = input('Enter time to reach destination: ');
Abnd = input('Enter the acceleration bound of drone: ');
b = input('Enter upper bound for value taken by parameter B: ');
W1 = input('Enter first corner of Window: ');
W2 = input('Enter second corner of Window: ');
W3 = input('Enter third corner of Window: ');
W4 = input('Enter fourth corner of Window: ');
R = input('Enter the influence radius of drone: ');

W = [W1;W2;W3;W4];

S = compileSet(i,f,W,0,tf,e,Abnd,b,R);

createGraph(i,f,tf,S,W);