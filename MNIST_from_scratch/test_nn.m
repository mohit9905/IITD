t1=(tst_images(:,970));
t1=reshape(t1,[28 28]);
figure
imshow(t1)
t1=reshape(t1,[784 1]);
[hidden_layer_state, output_layer_state] = forward_propagation...
    (t1, model.input_to_hidden_weights, model.hidden_to_output_weights,...
     model.hidden_bias, model.output_bias);
 
 output_layer_state