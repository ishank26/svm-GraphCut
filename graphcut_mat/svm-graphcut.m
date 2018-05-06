oimg = imread('./results/bridge.jpg') %f1
addpath('./maxflow');
%img = imread('./results/village.jpg'); %f2

oimg = imresize(oimg, 0.2);
img = imgaussfilt(oimg,2)
[r, c, d] = size(img);
n = r*c;
imshow(img);
fg1 = imfreehand(gca);
fgroi1 = sparse(createMask(fg1));
fg2 = imfreehand(gca);
fgroi2 = sparse(createMask(fg2));
bg1 = imfreehand(gca);
bgroi1 = sparse(createMask(bg1));
bg2 = imfreehand(gca);
bgroi2 = sparse(createMask(bg2));

fgroi = fgroi1+fgroi2

bgroi = bgroi1+bgroi2

%fgroi = fgroi1
%bgroi = bgroi2

[fy, fx] = find(fgroi);
[by, bx] = find(bgroi);



%[fx fy] = ginput(180) % x-column, y-row
%[bx by] = ginput(180)

%load('runproj.mat');


X = featx(fx, fy, bx, by, img);
Y = featy(fx, bx);


disp(X);
sigmasq = 1000;
C = 100;


model =  fitcsvm(X,Y,'KernelFunction','rbf','BoxConstraint',C,'KernelScale',sqrt(2000), 'solver', 'L1QP', 'classNames', [0,1]); % kenel scale = sigma
[preds, score]= predict(model, X);

[svmpred, svmscore, mu1, std1, mu0, std0] = makescore(score);

trainacc = sum(Y==preds)/length(Y);

[scoreSVMmodel,scoreParameters] = fitSVMPosterior(model);
scoreTransform = scoreSVMmodel.ScoreTransform;

disp("SVM training complete");
E = edges4connected(r,c);
u= E(:,1);
v = E(:,2);

%%%%%%%%%%%%%%%%
lambda = 0.2;
%%%%%%%%%%%%%%%%
disp("Making Graph");
% Boundary penalty U2 

U2 = makeU2(img, u, v);
U2 = U2.*(1-lambda);
% A = nxn
A = sparse(u, v, U2, n, n);

% T = nx2, T(,1) = source, T(,2) = sink
% for all pixels compute U1
T = makeT(img, model, u, v, mu1, mu0);
T= lambda*T;


% add constraints
varmat = img2var(img);

totalU2neigh = zeros(1,n);
for i = 1:length(n)
    pixlid = i;
    cpx = find(u==pixlid);
    disp(i)
    totalU2neigh(1,i) = neighU2(cpx,U2);
end
maxwt = max(totalU2neigh);
disp(maxwt);
for i = 1:length(Y)
    cls = Y(i,1);
    tmpc = [fx; bx]; % column
    tmpr = [fy; by]; % row
    col = uint16(tmpc(i,1));
    row = uint16(tmpr(i,1));
    pixlid = varmat(row,col);
    if cls == 1
        % foreground
        T(pixlid,1) = 1 + maxwt;
        T(pixlid,2) = 0;
    else
        % background
        T(pixlid,1) = 0;
        T(pixlid,2) = 1 + maxwt;
    end
end



T = sparse(T);


tic
[flow, labels] = maxflow(A,T);
toc


label = reshape(labels,[r, c]);

mask = ~label;


mask = cat(3, mask, mask, mask);


filter = uint8(oimg).^uint8(mask);

imshow(filter)
%%%%%%%%%%%%%%%%
% Util functions
%%%%%%%%%%%%%%%%


function f = neighU2(cpx, U2)
tmp = 0;
for j = 1:length(cpx)
    idx = cpx(j);
    tmp = tmp + U2(idx);
end
f = tmp;
end


% make feature vec X
function f = featx(fx, fy, bx, by, img)
tmpc = [fx; bx]; % column
tmpr = [fy; by]; % row
fvf = zeros(length(tmpr), 15); % c= 3*1 + n = 4*3
for i = 1:length(tmpr)
    r = int8(tmpr(i));
    c = int8(tmpc(i));
    fvf(i,:) = [img(r,c,1) img(r,c,2) img(r,c,3) img(r, c-1, 1) img(r, c-1, 2) img(r, c-1, 3) img(r-1,c,1) img(r-1,c,2) img(r-1,c,3) img(r,c+1,1) img(r,c+1,2) img(r,c+1,3) img(r+1,c,1) img(r+1,c,2) img(r+1,c,3)];
end
f = fvf;
end

% make label vector Y
function f = featy(fx, bx) 
    y1 = ones(length(fx),1);
    y0 = 0*ones(length(bx),1);
    f = [y1; y0];
end 

% Compute RBF kernel for features
function f = makeRBF(X, Z)
sigmasq = 1000;
m = length(X);
n = length(Z);
K = zeros(m,n);
for i = 1:m
    for j = 1:n
        K(i,j) = exp(-(norm(X(i,:) - Z(j,:))^2)/(2* sigmasq));
    end
end
end

function f = svmData(z, alp, C)
z = z(alp > 0, :);
z = z(C > alp, :);
f = z;
end

function f = makepred(K,palp,py,b)
preds = K*(palp.*py) + b;
f = preds;
end

function [svmpred, svmscore, mu1, std1, mu0, std0] = makescore(score)
svmpred = zeros(length(score), 1);
svmscore = zeros(length(score), 1);
for i = 1:length(score)
    [val, idx] = max(score(i,:));
    svmpred(i,:) = idx;
    svmscore(i,:) = val;
end
% convert to fb = 1, bg = 0,  class labels
svmpred(svmpred==1) = 0 ;
svmpred(svmpred==2) = 1 ;

J1 = [];
J0 = [];


for i = 1:length(svmpred)
    if svmpred(i) == 1
        J1 = [J1 svmscore(i)];
    else
        J0 = [J0 svmscore(i)];
    end
end


mu1 = mean(J1);
mu0 = mean(J0);
std1 = std(J1);
std0 = std(J0);

end


function f = makeU2(img, u,v)
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
sigma = 3;
%%%% RBF kernel %%%%%
%pixldiff = vecnorm(double([R(u) G(u) B(u)] - [R(v) G(v) B(v)])).^2;
%disp(size(pixldiff))
%pixldiff = pixldiff/(2*(sigma^2));
%U2 = exp(-pixldiff);

%%%% l1 kernel %%%%
pixldiff = vecnorm(double([R(u) G(u) B(u)] - [R(v) G(v) B(v)]), 1, 2);
pixldiff = pixldiff+1;
U2 = pixldiff.^-1;
f= U2;
end


function f = makeT(img, model, u, v, mu1, mu0)
[r c d] = size(img);
n = r*c;
T = zeros(n,2);
for i = 1:n
    pixlid = i;
    tmpfeat = makesvmx(pixlid,img,u,v);
    %disp(size(tmpfeat));
    [labl, sc] = predict(model, double(tmpfeat));
    scr = max(sc);
    if labl == 1
        %C= -6;
        %U1 = 1/1+exp(C*a1*scr+b);
        U1 = 1/(1+exp(-4 *scr/mu1));
        if U1 > 1
            disp(U1);
        end
        T(i,1) = U1; % alpha = 0, bg cost
        T(i,2) = 1-U1; % alpha = 1, fg cost
    else
        % pred == 0
        %C = 6;
        U1 = 1/(1+exp(-4 *scr/mu0));
        if U1 > 1
            disp(U1);
        end
        T(i,2) = U1; % alpha = 1, fg cost
        T(i,1) = 1-U1; % alpha = 0, bg cost
    end
end
f = T;
end

function f = img2var(img)
[r, c, d] = size(img);
n = r*c;
tmp = 1:n;
varmat = reshape(tmp,[r,c]);
f = varmat;
end

function f = makesvmx(pixlid,img,u,v)
% idx of center pixel in u
pixlid = int16(pixlid);
cpx = find(u==pixlid);
% neigh ids array
npx = v(cpx);
pivotR = img(:,:,1);
pivotG = img(:,:,2);
pivotB = img(:,:,3);
feat = [pivotR(pixlid) pivotG(pixlid) pivotB(pixlid)];
for i =1:length(npx)
    neighR = img(:,:,1);
    neighG = img(:,:,2);
    neighB = img(:,:,3);
    feat = [feat neighR(npx(i)) neighG(npx(i)) neighB(npx(i))];
end
if length(feat) < 15
    ls = (15 - length(feat))/3;
    for i = 1:ls
        feat = [feat pivotR(pixlid) pivotG(pixlid) pivotB(pixlid)]
    end
end
f = feat;
end

%%%%% CVX svm code %%%%%

%K = makRBF(X, X, sigmasq)
%load('runproj.mat')


%{
n = length(Y)
cvx_begin
    variables alp(n)
    minimize (0.5.*quad_form(Y.*alp,K) - ones(n,1)'*alp);
    subject to 
        alp <= C
        alp >= 0
        Y'*alp == 0
cvx_end

save('runproj.mat')



%load('runproj.mat')
px = svmData(X, alp, C)
py = svmData(Y, alp, C)
palp = svmData(alp, alp, C)
Q = makRBF(X, px, sigmasq)
n = length(X)
m = length(palp)
b = mean(Y-Q*(palp.*py))

preds = predict(Q,palp,py,b) % w'phi(x)+b
fpreds = preds
fpreds(fpreds > 0) = 1

acc = sum(Y==fpreds)/length(Y)
%}
