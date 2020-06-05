load('Trained_Network.mat');
input=[0 1 1;];
transposed_Input=input';
Weighted_sum=Weight*transposed_Input;
output=sigmoid_fun(Weighted_sum)
