input=[0 0 0;
       0 0 1;
       0 1 0;
       0 1 1;
       1 0 0;
       1 0 1;
       1 1 0;
       1 1 1;
       0 1 0;
       ];
correct_Output=[0
                1
                1
                1
                1
                1
                1
                1
                1
                ];
          
 Weight=rand(1,3);
 for iteration=1:100000
     Weight=SGD_method(Weight, input, correct_Output);
 end
 
 save('trained_Network.mat');

   