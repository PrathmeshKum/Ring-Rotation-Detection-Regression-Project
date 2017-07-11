%Feature Selection

%@Zhaozheng Yin, spring 2017

clc; clear all; %close all;
dir_training = 'training\';
dir_testing = 'testing\';
directory=char(pwd);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=cputime;
files = dir([dir_training '*.jpg']);
train_image_matrix = []; w_train = [];
train_img_num=size(files,1);

for i = 1:train_img_num
    filename = files(i).name;
    w_train = [w_train; str2double(filename(1:4))];
    im = imread([dir_training filename]);
    im = im(:,:,1);
    train_image_matrix = [train_image_matrix im(:)];
end
train_image_matrix = double(train_image_matrix); %every column in X is one vectorized input image
train_image_matrix = [ones(1,size(train_image_matrix,2)); train_image_matrix];

%train_image_matrix1=normalizeIm1(train_image_matrix,train_img_num+1);

mean_train=zeros(10202,1);

for iFile = 1:train_img_num;
    
    mean_train=mean_train+train_image_matrix(:,iFile);

end

mean_train=mean_train/train_img_num;
disp('Mean Calculated');

covar_train=zeros(10202,10202);

for iFile = 1:train_img_num;
    
     a=train_image_matrix(:,iFile)- mean_train;
     b=(transpose(a));     
     train_covar_numerator=a*b;
     covar_train=covar_train+train_covar_numerator;
end    

covar_train=covar_train/train_img_num;
disp('Covariance Calculated');


features=0;
feature_num_matrix= [];
c=zeros(1,10202);

for iFile = 1:10202;
    
    d=covar_train(iFile,:);
    
    if d==c;
        
        features=features+1;
        feature_num_matrix(features,1)=iFile;
        
    else
        continue
    end
end

% Deleting the 'zero' variation rows from the image matrix

[train_image_matrix_new,ps] = removerows(train_image_matrix,'ind',feature_num_matrix');

% Computation of phi_hat

A=train_image_matrix_new*train_image_matrix_new';
[~,p]=chol(A);
Anew=nearestSPD(A);
[~,p1]=chol(Anew);
B=train_image_matrix_new*w_train;
phi_hat=(Anew)\(B);

% Creating testing image matrix


files = dir([dir_testing '*.jpg']);
test_image_matrix = []; w_test = [];
test_img_num=size(files,1);

for i = 1:test_img_num
    filename = files(i).name;
    w_test = [w_test; str2double(filename(1:4))];
    im = imread([dir_testing filename]);
    im = im(:,:,1);
    test_image_matrix = [test_image_matrix im(:)];
end
test_image_matrix = double(test_image_matrix); %every column in X is one vectorized input image
test_image_matrix = [ones(1,size(test_image_matrix,2)); test_image_matrix];

%test_image_matrix1=normalizeIm1(test_image_matrix,test_img_num+1);

[test_image_matrix_new,ps1] = removerows(test_image_matrix,'ind',feature_num_matrix');

% INFERENCE ALGORITHM

w_test_hat = phi_hat'*test_image_matrix_new;
w_test_hat = w_test_hat';
w_test_num = size(w_test,1);
w_test_gt = w_test;
w_test_hat_rot = w_test_hat;

for i = 1:w_test_num
    
    w_test_gt(i,1) = w_test(i,1)/(3.14*pi);
    w_test_hat_rot(i,1) = w_test_hat(i,1)/(3.14*pi);
    
end


error=w_test_hat_rot-w_test_gt;
mod_error=abs(error);
deviation=sum(mod_error)/w_test_num;

disp(['file execution time: ' num2str(cputime-tt)]);

% Visualization

plot(w_test_gt);
hold on;
plot(w_test_hat_rot);
xlabel(' Image Number ');
ylabel(' Rotation angles ');
title('Deviation of predicted rotation angles from ground truth ');
grid on;

path=[directory '\feature_sel.mat'];
save(path);

Z=zeros(10202,1);

for i=(1:10202)
    
    m=train_image_matrix(i,:);
    n=var(m);
    Z(i,1)=n;
    
end