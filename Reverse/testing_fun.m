load('Trained_Network.mat');
input=[1 1 0;];
transposed_Input=input';
Weighted_sum=Weight*transposed_Input;
output=sigmoid_fun(Weighted_sum)
