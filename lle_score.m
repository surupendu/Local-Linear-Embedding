%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate LLE_Score of Image
%Input:Image Matrix and no of K nearest Neighbours
%Output: Matrix Y containing reduced dimension of each image in I1
%Authors: Surupendu Gangopadhyay and Abhijeet Ghodgaonkar
%Date: 30/3/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[Y1] = lle(I1,k)

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

[row1,col1] = size(Y);

%%Find K nearest neighbours
i = 1:row1;
[Idx1(i,:),D1] = knnsearch(Y,Y(i,:),'K',k+1);
%Idx contains the indices of k nearest neighbours
%%End of Calculation

%%Construct Weight Matrix
M = zeros(row1);
[r1,c1] = size(Idx1);
e1 = ones(c1-1,1);
% W is the weight matrix containing the 
% weights of the nearest neighbours

gamma = 10^-5;
Id = eye(k);
for i=1:row1
    V1 = [];
    j=2:c1;
    V1(j-1,:) = Y(Idx1(i,j),:);
    V1 = V1';
    X = Y(i  ,:)';
    G = (X*e1' - V1)' * (X*e1' - V1);
    G = (G+(gamma*Id))^(-1);
    Mij = (e1'*G)/(e1'*G*e1);
    %j=2:c;
    M(i,Idx1(i,j)) = Mij(1,j-1);
end
%End of calculation

%Feature Selection Step
X = W - M;
[r2,c2] = size(X);

for i=1:c2
    X1(:,i) = norm(X(:,i),'fro');
end

[B,I] = sort(X1,'ascend');

%Reselect Features as per score
for i=1:k
    j = I(:,i);
    Y1(:,i) = V(:,j);
end