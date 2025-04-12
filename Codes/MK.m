function [g,h,A,B,C,D] = MK(mu,sigma)
one = ones(size(sigma,1),1);
A = one'*(sigma\mu);
B = mu'*(sigma\mu);
C = one'*(sigma\one);
D = B*C-A^2;

g = (B*(sigma\one)-A*(sigma\mu))/D;
h = (C*(sigma\mu)-A*(sigma\one))/D;
end
