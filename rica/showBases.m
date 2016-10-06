function [out] = showBases(theta,iterationType,i,funEvals,f,t,gtd,g,d,optCond,varargin)
global params
%fprintf('size(theta)\n');
%size(theta)
%fprintf('params.numFeatures = %d\n',params.numFeatures);
%fprintf('params.n = %d\n',params.n);
if mod(i,2) == 0    
    W = reshape(theta, params.numFeatures, params.n);
    display_network(W');
end
out = 0;
