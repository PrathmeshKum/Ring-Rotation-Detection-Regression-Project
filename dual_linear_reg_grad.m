% Dual Linear Regression For Gradient Domain

%@Zhaozheng Yin, spring 2017

clc; clear all; %close all;
dir_training = 'training\';
dir_testing = 'testing\';

training_files = dir(dir_training);
testing_files = dir(dir_testing);


directory=char(pwd);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=cputime;
files = dir([dir_training '*.jpg']);
w_train = [];
%train_img_num=size(files,1);

[train_image_matrix,train_img_num]=img_grad(dir_training,training_files);
train_image_matrix = [ones(1,size(train_image_matrix,2)); train_image_matrix];

train_img_num=train_img_num-1;

for i = 1:train_img_num
    
    filename = files(i).name;
    w_train = [w_train; str2double(filename(1:4))];
    im = imread([dir_training filename]);
    im = im(:,:,1);

end

%train_image_matrix1=normalizeIm1(train_image_matrix,train_img_num+1);

A=train_image_matrix'*train_image_matrix;
[~,p]=chol(A);
Anew=nearestSPD(A);
[~,p1]=chol(Anew);

A1=train_image_matrix*train_image_matrix';
B=train_image_matrix*w_train;

psi_hat=(Anew)\(w_train);             % psi for non-regularized
phi_hat=train_image_matrix*psi_hat;   % phi for non-regularized

term=w_train-(train_image_matrix')*phi_hat;
sig=((term')*term)/train_img_num; % Variance for non regularized

lambda=200; % change this value
var_prior=sig/lambda; % variance of prior

Anew_1=A1+((sig/var_prior)*eye(size(train_image_matrix,1)));   % best found deviation value at lambda = 200
phi_hat_reg=(Anew_1)\(B);

disp('Phi Hat Calculated');


% Creating testing image matrix

files = dir([dir_testing '*.jpg']);
w_test = [];
%test_img_num=size(files,1);

[test_image_matrix,test_img_num]=img_grad(dir_testing,testing_files);
test_image_matrix = [ones(1,size(test_image_matrix,2)); test_image_matrix];

test_img_num=test_img_num-1;



for i = 1:test_img_num
    
    filename = files(i).name;
    w_test = [w_test; str2double(filename(1:4))];
    im = imread([dir_testing filename]);
    im = im(:,:,1);
   
end

%test_image_matrix1=normalizeIm1(test_image_matrix,test_img_num+1);

% INFERENCE ALGORITHM

w_test_hat = phi_hat_reg'*test_image_matrix;
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

path=[directory '\dual_linear_reg_grad.mat'];
save(path);