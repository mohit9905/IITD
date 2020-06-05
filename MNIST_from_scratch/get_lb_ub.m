function[lb,ub,lb1,ub1]= get_lb_ub(input_to_hidden_weights,...
    hidden_to_output_weights, hidden_bias, output_bias);

train_images = loadMNISTImages('train-images.idx3-ubyte'); % initialize figure  

input=train_images(:,1);
inputs_to_hid_units = input_to_hidden_weights' * input+hidden_bias ;

%hidden_layer_state1 = 1 ./ (1 + exp(-inputs_to_hid_units));   %sigmoid activation function
hidden_layer_state1=tansig(inputs_to_hid_units);
lb=hidden_layer_state1;
ub=hidden_layer_state1;

inputs_to_softmax1 = hidden_to_output_weights' * hidden_layer_state1+output_bias;          

lb1=input;
ub1=input;

%output_layer_state = logsig(inputs_to_softmax)
count=0;

for i=1:10000
input=train_images(:,i);
count=count+1;
for j=1:784

    if(input(j)<lb1(j))
        lb1(j)=input(j);
    end
    if(input(j)>ub1(j))
        ub1(j)=input(j);
    end
end

inputs_to_hid_units = input_to_hidden_weights' * input+hidden_bias ;
hidden_layer_state1=tansig(inputs_to_hid_units);
for k=1:128
    
    if(hidden_layer_state1(k)<lb(k))
        lb(k)=hidden_layer_state1(k);
    end
    if(hidden_layer_state1(k)>ub(k))
        ub(k)=hidden_layer_state1(k);
    end
end
end
end
                  

                    
                   
%output_layer_state = exp(inputs_to_softmax);



% normalizing for prop. distr.
% Softmax function
%output_layer_state = output_layer_state ./ ...
         %            repmat(sum(output_layer_state, 1), output_size, 1)

                 





