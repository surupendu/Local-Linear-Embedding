%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate LLE of Image
%Input:Image Matrix and no of K nearest Neighbours
%Output: Matrix Y containing reduced dimension of each image in I1
%Authors: Surupendu Gangopadhyay and Abhijeet Ghodgaonkar
%Date: 30/3/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Y] = lle(I1,k)

I1 = I1';
[row,col] = size(I1);

%%Find K nearest neighbours
i = 1:row;
[Idx(i,:),D] = knnsearch(I1,I1(i,:),'K',k+1);
%Idx contains the indices of k nearest neighbours
%%End of Caluculation


[r,c] = size(Idx);

%%Construct Weight Matrix
W = zeros(row);
[r,c] = size(Idx);
e = ones(c-1,1);
% W is the weight matrix containing the 
% weights of the nearest neighbours
for i=1:row
    V = [];
    j=2:c;
    V(j-1,:) = I1(Idx(i,j),:);
    V = V';
    X = I1(i  ,:)';
    G = (X*e' - V)' * (X*e' - V);
    G = G^(-1);
    Wij = (e'*G)/(e'*G*e);
    %j=2:c;
    W(i,Idx(i,j)) = Wij(1,j-1);
end


%End of calculation

%Construction of Matrix Y in lower Dimension
%Y contains the reduced form of the image
[r1,c1] = size(W);
I = eye(r1);

M = (I-W)' * (I-W);

[V,D] = eig(M);

q = 5;

i=2:q+1;
Y = [];
%Each row of Y corresponds to the the row of I1
Y(:,i-1) = V(:,i);


