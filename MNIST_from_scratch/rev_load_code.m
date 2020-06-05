
function[inputs_to_softmax,inputs_to_hid_units,hidden_layer_state]=rev_load_code(input_to_hidden_weights,...
    hidden_to_output_weights, hidden_bias, output_bias);

tst_images=loadMNISTImages('t10k-images.idx3-ubyte');

in=tst_images(:,41);
in=reshape(in,[28 28]);
figure
imshow(in);
in=reshape(in,[784 1]);



%Hidden layer states
inputs_to_hid_units = input_to_hidden_weights' * in+hidden_bias 
                      
                  
%hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hid_units));   %sigmoid activation function
hidden_layer_state=tansig(inputs_to_hid_units);


figure
imshow(reshape(hidden_layer_state,[16,8]));

inputs_to_softmax = hidden_to_output_weights' * hidden_layer_state+output_bias;          
                    
                   
output_layer_state = tansig(inputs_to_softmax);
end


                 





