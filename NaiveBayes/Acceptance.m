function [ Best_Solution,Xt,Index,sortNeighbours,IX,accuracyXb,AccuracyXt,Best_Accuracy,Xb] = Acceptance(Neighbors,label_vector, instance_matrix,nDataset,mDataset,Xb,Tc,label_idx,classifier);
[b,z]=size(Neighbors);
[mm,n] = size(Neighbors);
%% Evaluate Xb
k =10; %% Number of folds for K-Fold Cross Validation
cp = cvpartition(label_idx ,'k',10);
t =1;
while (t < (b+1))
Neighborssingle (1,1:n) = Neighbors(t,1:n);
instance_matrix_Xt= instance_matrix;
instance_matrix_Xt(:,(Neighborssingle(1,:) == 0)) = [];
%% Applying K-fold Cross Validation on Xt
for K = 1:k
   %% Training Set
   selected_train_set = training(cp,K);
   y = 1;
   for j = 1:mDataset
       if selected_train_set(j,1) == 1
           label_vector_train_Xt(y,1) = label_vector(j,1);
           instance_matrix_train_Xt(y,:) = instance_matrix_Xt (j,:);
           y = y +1;
       end
   end   
 %% Tarin the model
if classifier == 1
    modelTrain =  NaiveBayes.fit(instance_matrix_train_Xt,label_vector_train_Xt);
else
    modelTrain =  NaiveBayes.fit(instance_matrix_train_Xt,label_vector_train_Xt,'Distribution', 'kernel');
end
 
%% Testing Set
selected_test_set = test(cp,K);
y = 1;
for j = 1:mDataset
       if selected_test_set (j,1) == 1
           label_vector_test_Xt(y,1) = label_vector(j,1);
           instance_matrix_test_Xt(y,:) = instance_matrix_Xt (j,:);
           y = y +1;
       end
end   
%% Test the model
predictLabels = predict(modelTrain,instance_matrix_test_Xt);
ConfusionMat = confusionmat(label_vector_test_Xt,predictLabels);
accuracyXt_KFold(1,K) = (sum(diag(ConfusionMat))/ sum(sum(ConfusionMat))) * 100;

end
accuracyXt(1,t) = sum (accuracyXt_KFold)/K;
t=t+1;
end
 [sortNeighbours,IX] = sort(  accuracyXt, 'descend');
 AccuracyXt = sortNeighbours(1,1);
 Index= IX(1,1);
 Xt= Neighbors (Index,:);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
 %% Accuracy of Xb
k=10; 
cp = cvpartition(label_idx ,'k',10); 
instance_matrix_Xb= instance_matrix;
instance_matrix_Xb(:,(Xb(1,:) == 0)) = []; 

%% Applying K-fold Cross Validation on Xb
for K = 1:k
   %% Training Set
   selected_train_set = training(cp,K);
   y = 1;
   for j = 1:mDataset
       if selected_train_set(j,1) == 1
           label_vector_train_Xb(y,1) = label_vector(j,1);
           instance_matrix_train_Xb(y,:) = instance_matrix_Xb (j,:);
           y = y +1;
       end
   end
     
 %% Tarin the model
if classifier == 1
    modelTrain =  NaiveBayes.fit(instance_matrix_train_Xb,label_vector_train_Xb);
else
    modelTrain =  NaiveBayes.fit(instance_matrix_train_Xb,label_vector_train_Xb,'Distribution', 'kernel');
end
%% Testing Set
selected_test_set = test(cp,K);
y = 1;
for j = 1:mDataset
       if selected_test_set (j,1) == 1
           label_vector_test_Xb(y,1) = label_vector(j,1);
           instance_matrix_test_Xb(y,:) = instance_matrix_Xb (j,:);
           y = y +1;
       end
end
%% Test the model
predictLabels = predict(modelTrain,instance_matrix_test_Xb);
ConfusionMat = confusionmat(label_vector_test_Xb,predictLabels);
accuracyXb_KFold(1,K) = (sum(diag(ConfusionMat))/ sum(sum(ConfusionMat))) * 100;
end
accuracyXb = sum (accuracyXb_KFold)/K;
%% Checking the Acceptance Criteria 
[s,z]=size(Neighbors);
for i=1:s
     p=rand;
     if (accuracyXt(1,i)> accuracyXb(1,1))
         Xb = Neighbors(i,:);  
     elseif  (accuracyXt(1,i) == accuracyXb(1,1))
      Xb = Xb;
     elseif ( (accuracyXt(1,i) < accuracyXb(1,1)) && ( p < exp ((-(accuracyXb(1,1)- accuracyXt(1,i)))/Tc)))
        Xb = Neighbors(i,:);
      
     end    
end   
 Best_Accuracy(1,1)= max(accuracyXt, accuracyXb);
 if (accuracyXt > accuracyXb)
     Best_Solution (1,:) = Xt(1,:);
      Best_Accuracy(1,1) = accuracyXb; 
 
 elseif  (accuracyXt== accuracyXb)
     Best_Solution (1,:) = Xb(1,:); 
      Best_Accuracy(1,1) = accuracyXb;
 elseif  (accuracyXt< accuracyXb)
     Best_Solution (1,:) = Xb(1,:); 
     Best_Accuracy(1,1) = accuracyXb;
 end
 Xb (1,:)= Best_Solution(1,:);
   end

  

