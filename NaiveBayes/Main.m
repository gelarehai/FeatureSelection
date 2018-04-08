close all; clear all; clc
%--------------------------------------------
%% Dataset (Un-comment a desired dataset)
%% Heart Dataset : Two classes 
input = genvarname('heart');  
%% Cleveland Dataset : Five classes 
%input = genvarname('cleveland'); 
%% Ionosphere Dataset : Two classes 
%input = genvarname('ionosphere'); 
%% Water Dataset : Three classes 
%input = genvarname('water'); 
%% Waveform Dataset : Two classes 
%input = genvarname('waveform'); 
%% Sonar Dataset : Two classes 
%input = genvarname('sonar'); 
%% Handwritte Dataset : Ten classes 
%input = genvarname('handwritten'); 
%% Arrhythmia Dataset : Sixteen classes 
%input = genvarname('arrhythmia'); 
%% Libras Dataset : Fifteen classes 
%input = genvarname('libras');
%% Secom Dataset : Two classes
%input = genvarname('secom');
%% Mfet Dataset : Two classes
%input = genvarname('Mfeat');
%% Ozone Dataset : Two classes
%input = genvarname('ozone');
if strcmpi(input, 'heart')
       Dataset = 'DataSet/heart.csv'
       filter_class = 2; 
       classifier = 2;
elseif strcmpi(input,'cleveland')
       Dataset = 'DataSet/cleveland.csv'
       filter_class = 0;
       classifier = 2;
elseif strcmpi(input,'ionosphere')
        Dataset = 'DataSet/ionosphere.csv'
        filter_class = 1;
        classifier = 2;
elseif strcmpi(input,'water')
        Dataset = 'DataSet/water.csv'
        filter_class = 1;
        classifier = 2;  
elseif strcmpi(input,'waveform')
        Dataset = 'DataSet/waveform.csv'
        filter_class = 1;
        classifier = 2;
elseif strcmpi(input,'sonar')
        Dataset = 'DataSet/sonar.csv'
        filter_class = 1;
        classifier = 2 ; 
elseif strcmpi(input,'handwritten')
        Dataset = 'DataSet/handwritten.csv'
        filter_class = 1;
        classifier = 2; 
elseif strcmpi(input,'arrhythmia')
        Dataset = 'DataSet/arrhythmia.csv'
        filter_class = 1;
        classifier = 2 ;
elseif strcmpi(input,'libras')
        Dataset = 'DataSet/libras.csv'
        filter_class = 1;
        classifier = 1; 
elseif strcmpi(input,'secom')
        Dataset = 'DataSet/secom.csv'
        filter_class = 1;
        classifier = 2; 
        % Imputation
        Dataset = knnimpute(Dataset)
elseif strcmpi(input,'Mfeat')
        Dataset = 'DataSet/Mfeat.csv'
        filter_class = 1;
        classifier = 2;
elseif strcmpi(input,'ozone')
        Dataset = 'DataSet/ozone.csv'
        filter_class = 1;
        classifier = 1; 
end
Dataset = csvread(Dataset);
[mDataset,nDataset] = size(Dataset);
%% Creating an initial solution (Xb)
[Xb] = randi([0 1], 1,(nDataset-1));
%% Parameter Setting
Tc = 1; %Initiial Tempreture 
alpha = 0.003 ; % Degradating rate of Tc
iteration = 5000; 
%--------------------------------------------
%% Reading the Dataset
[mDataset,nDataset] = size(Dataset);
[label] = Dataset(:,nDataset);
[matrix] =Dataset(:,1:(nDataset-1)) ;
[mDataset,nDataset] = size(Dataset);
for i =1:mDataset
  label_idx (i,1) = i;
  label_vector(i,1) = label(i,1);
  instance_matrix(i,1:(nDataset-1))= matrix(i,1:(nDataset-1));  
end
%% Using a Filter Approach to rank the features
BC = label_vector ==  filter_class;
[IDX, Feature_Score] = rankfeatures(instance_matrix',BC);
%% Sorting the features in an non-decreasing order
[F,IIX]= sort (Feature_Score,'ascend');
%% WFLNS Main Loop
for iter = 1:iteration      
Tc = alpha* Tc;
%% Repair and Destruction 
[Xt,Neighbors] = RepairDestroyed(Xb,F,IIX,iter,nDataset);
%% Acceptance Crtiteria
[Best_Solution,Xt,Index,sortNeighbours,IX,accuracyXb,AccuracyXt,Best_Accuracy,Xb] = Acceptance(Neighbors,label_vector, instance_matrix,nDataset,mDataset,Xb,Tc,label_idx,classifier);
answer (iter,:)= Best_Accuracy(1,1);
Best_Solutions (iter,:) = Best_Solution(1,:);
SelectedFeatures(iter,1) = sum (Best_Solution(1,:));
iter
end
 FinaSubset = Best_Solutions(iter,:); 
%% Final Accuracy
 FinalAsnwer = answer(iter,1);

