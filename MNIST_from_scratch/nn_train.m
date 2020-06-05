%Parameter
batchsize = 50;    % Mini-batch size.
learning_rate0 = 0.02;
momentum = 0.0;    % Momentum
numhid = 128;       % hidden layer size
epochs = 50;
lambda = 0.00; % L2 regularization
c=0.01
%dropout = 0;        % activating dropout

%Load MNIST

train_images = loadMNISTImages('train-images.idx3-ubyte'); % initialize figure  
train_labels = loadMNISTLabels('train-labels.idx1-ubyte'); % initialize figure
tst_images=loadMNISTImages('t10k-images.idx3-ubyte');
tst_labels=loadMNISTLabels('t10k-labels.idx1-ubyte');

train_images=reshape(train_images,[784,200,300]);
train_labels=reshape(train_labels,[1,200,300]);
[input_size, batchsize, numbatches]=size(train_images);

output_size = 10;


%INITIALIZATING VALUES

input_to_hidden_weights  = 1/sqrt(input_size) * randn(input_size, numhid);
hidden_to_output_weights =  randn(numhid, output_size);

hidden_bias = zeros(numhid,1);
output_bias = zeros(output_size,1);

CE_array = zeros(numbatches,epochs);   %cross entropy array

% initializing gradient matrices

input_to_hidden_weights_delta = zeros(input_size, numhid);
hidden_to_output_weights_delta = zeros(numhid, output_size);

hidden_bias_delta = zeros(numhid,1);
output_bias_delta = zeros(output_size,1);

% other
tiny = exp(-30);                        % avoiding log(0)
show_training_CE_after = 100;           % for tracking progress
%show_validation_CE_after = 100;         % for tracking progress
count = 0;

CE_array_avg = zeros(floor(numbatches/show_training_CE_after),epochs);
%CE_array_valid = zeros(floor(numbatches/show_validation_CE_after),epochs);


%LOOP OVER EPOCHS
for epoch = 1:epochs
  
  fprintf(1, 'Epoch %d\n', epoch);
  this_chunk_CE = 0;
  trainset_CE = 0;

  %SCHEDULING LEARNING RATE
learning_rate = learning_rate0; %/ (1+((epoch-1)/10)^2);
weight_dec = 1 - learning_rate * lambda / (batchsize * numbatches);
disp(' Learning rate :')
disp(learning_rate)

for m=1:numbatches      %loop over mini-batches

    % Forward propagate
    
input_batch = train_images(:,:,m);

[hidden_layer_state, output_layer_state,inputs_to_softmax] = forward_propagation...
    (input_batch, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
 
 %error function
 
expansion_matrix = eye(output_size);
target_batch     = train_labels(:,:,m);
target_vectors   = expansion_matrix(:,target_batch+1); % +1 avoiding zero indices
target_vectors(target_vectors==0)=-1;

% LOG Likelihood (Cross entropy) error function
CE = sum(sum((target_vectors-output_layer_state).*(target_vectors-output_layer_state)))/batchsize;
%CE=CE+sum(sum(c*(inputs_to_softmax).^2))/batchsize;
CE_array(m,epoch)=CE;

%show avg error
count =  count + 1;
this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
trainset_CE = trainset_CE + (CE - trainset_CE) / m;
%fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
if mod(m, show_training_CE_after) == 0
    fprintf(1, '\n');
    count = 0;
    CE_array_avg(m/show_training_CE_after,epoch) = this_chunk_CE;
    this_chunk_CE = 0;
end

%%BACKPROPAGATION

% Error fcn derivative
error_deriv = (output_layer_state - target_vectors).*(1-output_layer_state.*output_layer_state);
                                                %+2*0.01*(inputs_to_softmax);

% output layer
hid_to_output_weights_grad =  hidden_layer_state * error_deriv' +(2*c*hidden_layer_state*(inputs_to_softmax)');

output_bias_grad           = sum(error_deriv, 2) +sum(2*c*inputs_to_softmax,2);

% hidden layer
back_propagated_deriv = (hidden_to_output_weights *( error_deriv+2*c*inputs_to_softmax )) ...
       .* (1 - hidden_layer_state.* hidden_layer_state);

input_to_hid_weights_grad = input_batch * back_propagated_deriv';
hidden_bias_grad          = sum(back_propagated_deriv, 2);


%UPDATING WEIGHT AND BIAS

% input_to_hidden_weights
input_to_hidden_weights_delta = momentum .* input_to_hidden_weights_delta...
    + input_to_hid_weights_grad ./ batchsize;
input_to_hidden_weights = input_to_hidden_weights * weight_dec...
    - learning_rate * input_to_hidden_weights_delta;

% hidden_to_output_weights
hidden_to_output_weights_delta = momentum .* hidden_to_output_weights_delta...
    + hid_to_output_weights_grad ./ batchsize;
hidden_to_output_weights = hidden_to_output_weights * weight_dec...
    - learning_rate * hidden_to_output_weights_delta;

% hidden_bias
hidden_bias_delta = momentum .* hidden_bias_delta...
    + hidden_bias_grad ./ batchsize;
hidden_bias = hidden_bias - learning_rate * hidden_bias_delta;

% output_bias
output_bias_delta = momentum .* output_bias_delta...
    - output_bias_grad ./ batchsize;
output_bias = output_bias - learning_rate * output_bias_delta;

end

fprintf(1, '\rAverage Training CE %.3f\n', trainset_CE);



test_acc=accuracy_test(input_to_hidden_weights,hidden_to_output_weights,hidden_bias,output_bias,tst_images,tst_labels);
fprintf(1, '\rTest Acc %.3f\n', test_acc);
end

fprintf(1, 'Finished Training.\n');
fprintf(1, 'Final Training CE %.3f\n', trainset_CE);


%save model

model.input_to_hidden_weights  = input_to_hidden_weights;
model.hidden_to_output_weights = hidden_to_output_weights;
model.hidden_bias              = hidden_bias;
model.output_bias              = output_bias;

save model model

%plot_CE(CE_array,CE_array_avg,CE_array_valid)
%test_accuracy



