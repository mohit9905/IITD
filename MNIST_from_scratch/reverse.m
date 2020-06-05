
Output=[ 
   13.2625
    1.1093
    2.1043
   -2.3905
   -1.1752
    3.7085
    3.0320
    1.8024
    1.5672
    0.9495
  ];

%for i=1:10
 %   Output(i)=-log(1/(Output(i)+exp(-10)) - 1);
%end

[inputs_to_softmax,inputs_to_hid_units,hidden_layer_state]=rev_load_code(input_to_hidden_weights,...
    hidden_to_output_weights, hidden_bias, output_bias);

%lb=0.299*ones(1,100);          %lower bound
%ub=1*ones(1,100);           %upper bound
C=0.01;                        %C
epsilon=0.0001;                   %epsilon
Output=inputs_to_softmax;
x=Output-output_bias;            %subtracting bias value
Weighted_sum=x';                  


[lb,ub,lb1,ub1]=get_lb_ub(input_to_hidden_weights,...
  hidden_to_output_weights, hidden_bias, output_bias);

%1*100                                                                           
var=HighDimension_sir(Weighted_sum,hidden_to_output_weights',lb', ub', C, epsilon, 128);
%var=hidden_layer_state';
%figure
%imshow(reshape(hidden_layer_state,[10,10]));
%i=graythresh(var);
%var=imbinarize(var,i);
figure
imshow(reshape((var),[16,8]));
for r=1:128
var(r)= -log(1/(var(r)/2 + 1/2) - 1)/2;
end
%bias is 128*1

var=var-hidden_bias';
%lb1=-30*ones(784,1);
%ub1=ones(784,1); 
C1=0.01;
var=inputs_to_hid_units'-hidden_bias';
%1*784
inp=HighDimension_sir(var,input_to_hidden_weights', lb1', ub1', C1, epsilon, 784);
                                                                                                                                         

inp=reshape(inp,[28,28]);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
%inp=mat2gray(inp);
%I=graythresh(inp);
%inp=imbinarize(inp,I);
figure
imshow(inp)


