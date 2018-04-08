function [Xt,Neighbors] = RepairDestroyed(Xb,F,IIX,iter,nDataset);
IIX = IIX'; %% Sorted scores of features based on filter method
count= 1;
for i=1:nDataset-1
        Xb_selected_features (1,i) = IIX(1,i);
end
% Destruct lowest ranked part of the solution
k = randi([1 nDataset-1], 1,1); 
destruction_size = k;
% Recunstruct destroyed part of the solution
R = randi([1 k],1,1);
Neighbors = Xb;
for i=k:-1:1
   Neighbors(1,Xb_selected_features(1,i))=1;
end

for i=1:k-R
   Neighbors(1,Xb_selected_features(1,i))=0;
end
Summation =sum(Neighbors);
Neighbors;
Xt =  Neighbors;

end
    
    
   
    
    
    






 
 
 

    
    