load model
count=0;
 for i=1:size(tst_images,2)
     
     input=tst_images(:,i);
    [hidden_layer_state, output_layer_state] = forward_propagation...
    (input, input_to_hidden_weights, hidden_to_output_weights,...
     hidden_bias, output_bias);
  
 [prob,indices]=sort(output_layer_state,'descend');
 indices=indices-1;
 if(indices(1)==tst_labels(i))
     count=count+1;
 end
 end
 result=count/size(tst_images,2)
fprintf(1, '\n\nCorrectly classified images on test set: %.2f%% \n', result*100);
 
 
 
