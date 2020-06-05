load('trained_Network.mat')
out=.0409;
o=1/output;
Weight_sum=-(log(o-1));
transposed_inp=inv(Weight)*Weight_sum;
inp=transposed_inp';