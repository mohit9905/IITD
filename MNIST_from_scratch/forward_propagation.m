function [hidden_layer_state, output_layer_state,inputs_to_softmax] = ...
    forward_propagation(input_batch, input_to_hidden_weights,...
    hidden_to_output_weights, hidden_bias, output_bias)

[input_number, batch_size] = size(input_batch);
[numhid, output_size]      = size(hidden_to_output_weights);


%Hidden layer states
inputs_to_hid_units = input_to_hidden_weights' * input_batch...
                      + repmat(hidden_bias,1,batch_size);
                  
hidden_layer_state = (1 - exp(-inputs_to_hid_units)) ./ (1 + exp(-inputs_to_hid_units));   %sigmoid activation function

inputs_to_softmax = hidden_to_output_weights' * hidden_layer_state...            
                    + repmat(output_bias,1,batch_size);


% making softmax inputs =< 0
% inputs_to_softmax = inputs_to_softmax...
%                     - repmat(max(inputs_to_softmax), output_size, 1);

%tansig
output_layer_state = (1 - exp(-inputs_to_softmax)) ./ (1 + exp(-inputs_to_softmax));%exp(inputs_to_softmax);


% normalizing for prop. distr.
% Softmax function
% output_layer_state = output_layer_state ./ ...
%                      repmat(sum(output_layer_state, 1), output_size, 1);

                 
end




