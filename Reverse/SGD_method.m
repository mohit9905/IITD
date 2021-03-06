%Training By SGD method

function Weight=SGD_method(Weight,input,correct_Output);
alpha=0.9;

N=4;
for k=1:N
    transposed_input=input(k,:)';
    d=correct_Output(k);
    Weighted_sum1=Weight*transposed_input;
    Weighted_sum=sum(Weight_sum1);
    output=sigmoid_fun(Weighted_sum);
    
    error=d-output;
    delta=output*(1-output)*error;
    
    dweight=alpha*delta*transposed_input;
    
    Weight(1)=Weight(1)+dweight(1);
    Weight(2)=Weight(2)+dweight(2);
    Weight(3)=Weight(3)+dweight(3);
end
end

    