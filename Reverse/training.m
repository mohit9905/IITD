input=[0 0;
       0 1;
       1 0;
       1 1;
       ];
correct_Output=[0
                1 
                1 
                0];
          
 Weight=2*rand(2,2)-1;
 for epoch=1:1000
     Weight=SGD_method(Weight, input, correct_Output);
 end
 
 save('trained_Network.mat');

   