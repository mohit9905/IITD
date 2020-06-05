load('trained_Network.mat')
out=0.0076;
o=1/out;
Weight_sum=-log(o-1);
transposed_inp=pinv(Weight)*Weight_sum;
inp=transposed_inp'
