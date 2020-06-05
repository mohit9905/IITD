function [ datapoint_high ] = HighDimension( datapoint_low, G, lb, ub, C, epsilon, N )
%GETHIGHDIMENSIONDATASET
% This function maps a data point in the projected lower-dimensional
% subspace (K) to the original high dimensional subspace (N). The input
% parameters are:
% datapoint_low: The data point in K dimensions (1XK)
% G: The transformation matrix of size (KXN)
% lb: Lower bound - row vector of minimum of features in N-space (1XN).
% ub: Upper bound - row vector of maximum of features in N-space (1XN).
% C: scalar parameter determining tradeoff between approximation error and
% norm of solution vector.
% epsilon: scalar hyperparameter controlling tolerance of approximation
% N: scalar representing original high dimensional subspace (N)
% The function outputs the N-dimensional data point.
% Sumit Soman, 27  January 2014
%load('reverse1.m')

K=size(datapoint_low,2) ;

X=[randn(N,1); randn(K,1); randn(K,1)]; %[Z q+ q-]
f=[zeros(N,1); C*ones(K,1); C*ones(K,1)];

% Setup quadratic term of x'Qx

Q=[eye(N),zeros(N,K),zeros(N,K);...
    zeros(K,N),zeros(K,K),zeros(K,K);...
    zeros(K,N),zeros(K,K),zeros(K,K)];

% Setup linear constraints of Ax<=b

A=[G,eye(K),zeros(K,K);... %Gz - q+ <= x + eps
    -G, zeros(K,K),-eye(K)];%-Gz - q- <= -x + eps

eps_vector=epsilon*ones(K,1);
b=[datapoint_low'+eps_vector;-datapoint_low'+eps_vector];

% Setup lower and upper bounds
lower_bounds=[lb';zeros(K,1);zeros(K,1)];
upper_bounds=[ub';inf*ones(K,1);inf*ones(K,1)];
opts = optimset('Algorithm','interior-point-convex');


[X,fval,EXITFLAG] = quadprog(Q,f,A,b,[],[],lower_bounds,upper_bounds,X,opts);

if (EXITFLAG==1)
    datapoint_high=X(1:N)';
else
    fprintf(2,'Algorithm did not converge!');
    datapoint_high=rand(N,1);
end

end
