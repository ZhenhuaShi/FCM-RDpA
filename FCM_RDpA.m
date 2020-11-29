function [RMSEtrain,RMSEtest,CB,SigmaB,WB,fB]=FCM_RDpA(XTrain,yTrain,XTest,yTest,alpha,rr,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0)
% This function implements the FCM_RDpA algorithm in the following paper:
%
% Z. Shi, D. Wu, C. Guo, C. Zhao, Y. Cui and F.-Y. Wang, "FCM-RDpA: TSK Fuzzy Regression Model Construction 
% Using Fuzzy c-Means Clustering, Regularization, DropRule, and Powerball AdaBelief," IEEE Trans. on Fuzzy Systems, submitted, 2020.
%
% By Zhenhua Shi, zhenhuashi@hust.edu.cn
%
% %% Inputs:
% XTrain: N*M matrix of the training inputs. N is the number of samples, and M the feature dimensionality.
% yTrain: N*1 vector of the labels for XTrain
% XTest: {NValidation*M, NTest*M} matrix cell of the validation and test inputs 
% yTest: {NValidation*1, NTest*1} vector cell of the labels for XTest
% alpha: scalar, initial learning rate
% rr: scalar, L2 regularization coefficient 
% P: scalar, DropRule preservation rate
% gammaP: scalar, Powerball power exponent
% nRules: scalar, total number of rules
% nIt: scalar, maximum number of iterations
% Nbs: batch size. typically 32 or 64
% C0: nRules*M initialization matrix of the centers of the Gaussian MFs
% Sigma0: nRules*M initialization matrix of the standard deviations of the Gaussian MFs
% W0: nRules*(M+1) initialization matrix of the consequent parameters for the nRules rules
%
% %% Outputs:
% RMSEtrain: 1*nIt vector of the training RMSE at different iterations
% RMSEtest: {1*nIt, 1*nIt} vector cell of the validation and test RMSE at different iterations
% CB: nRules*M matrix of the centers of the Gaussian MFs
% SigmaB: nRules*M matrix of the standard deviations of the Gaussian MFs
% WB: nRules*(M+1) matrix of the consequent parameters for the nRules rules
% fB: 1*nRules vector of the firing levels for validation inputs

beta1=0.9; beta2=0.999; thre=inf;

if ~iscell(XTest)
    XTest={XTest};
    yTest={yTest};
end
[N,M]=size(XTrain);
Nbs=min(N,Nbs);
if nargin<12
    W0=zeros(nRules,M+1); % Rule consequents
    % FCM initialization
    [C0,U] = fcm(XTrain,nRules,[2 100 0.001 0]);
    Sigma0=C0;
    for r=1:nRules
        Sigma0(r,:)=std(XTrain,U(r,:));
        W0(r,1)=U(r,:)*yTrain/sum(U(r,:));
    end
    Sigma0(Sigma0==0)=mean(Sigma0(:));
end
C=C0; Sigma=Sigma0; W=W0;
minSigma=.1*min(Sigma0(:));

[CB,SigmaB,WB,fB]=deal(C,Sigma,W,zeros(1,nRules));
%% Iterative update
RMSEtrain=zeros(1,nIt); RMSEtest=cellfun(@(u)RMSEtrain,XTest,'UniformOutput',false);
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaC=zeros(nRules,M); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=zeros(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    
    for n=1:Nbs
        idsKeep=rand(1,nRules)<=P;
        f(n,idsKeep)=prod(exp(-(XTrain(idsTrain(n),:)-C(idsKeep,:)).^2./(2*Sigma(idsKeep,:).^2)),2);
        if sum(~isfinite(f(n,idsKeep)))
            continue;
        end
        if ~sum(f(n,idsKeep)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=prod(exp(-(XTrain(idsTrain(n),:)-C(idsKeep,:)).^2./(2*Sigma(idsKeep,:).^2)),2);
            idsKeep=true(1,nRules);
        end
        deltamuC=(XTrain(idsTrain(n),:)-C(idsKeep,:))./(Sigma(idsKeep,:).^2);
        deltamuSigma=(XTrain(idsTrain(n),:)-C(idsKeep,:)).^2./(Sigma(idsKeep,:).^3);
        fBar=f(n,idsKeep)/sum(f(n,idsKeep));
        yR=[1 XTrain(idsTrain(n),:)]*W(idsKeep,:)';
        yPred(n)=fBar*yR'; % prediction
        if isnan(yPred(n))
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        
        % Compute delta
        deltaYmu=(yPred(n)-yTrain(idsTrain(n)))*(yR*sum(f(n,idsKeep))-f(n,idsKeep)*yR')/sum(f(n,idsKeep))^2.*f(n,idsKeep);
        if ~sum(~isfinite(deltaYmu(:)))
            deltaC(idsKeep,:)=deltaC(idsKeep,:)+deltaYmu'.*deltamuC;
            deltaSigma(idsKeep,:)=deltaSigma(idsKeep,:)+deltaYmu'.*deltamuSigma;
            deltaW(idsKeep,:)=deltaW(idsKeep,:)+(yPred(n)-yTrain(idsTrain(n)))*fBar'*[1 XTrain(idsTrain(n),:)];
        end
    end
    
    % powerball
    deltaC=sign(deltaC).*(abs(deltaC).^gammaP);
    deltaSigma=sign(deltaSigma).*(abs(deltaSigma).^gammaP);
    deltaW=sign(deltaW).*(abs(deltaW).^gammaP);
    % AdaBelief
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*(deltaC-mC).^2;
    mCHat=mC/(1-beta1^it);
    vCHat=vC/(1-beta2^it);
    C=C-alpha*mCHat./(sqrt(vCHat)+10^(-8));
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*(deltaSigma-mSigma).^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    Sigma=max(minSigma,Sigma-alpha*mSigmaHat./(sqrt(vSigmaHat)+10^(-8)));
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*(deltaW-mW).^2;
    mWHat=mW/(1-beta1^it);
    vWHat=vW/(1-beta2^it);
    W=W-alpha*mWHat./(sqrt(vWHat)+10^(-8));
    
    % Training RMSE on the minibatch
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    for i=1:length(XTest)
        NTest=size(XTest{i},1);
        f=zeros(NTest,nRules); % firing level of rules
        for n=1:NTest
            f(n,:)=prod(exp(-(XTest{i}(n,:)-C).^2./(2*Sigma.^2)),2);
        end
        f(:,P==0)=0;
        yR=[ones(NTest,1) XTest{i}]*W';
        yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
        yPredTest(isnan(yPredTest))=nanmean(yPredTest);
        RMSEtest{i}(it)=sqrt((yTest{i}-yPredTest)'*(yTest{i}-yPredTest)/NTest);
        if isnan(RMSEtest{i}(it)) && it>1
            RMSEtest{i}(it)=RMSEtest{i}(it-1);
        end
        if nargout>2&&i==1&&RMSEtest{i}(it)<thre
            thre=RMSEtest{i}(it);
            [CB,SigmaB,WB,fB]=deal(C,Sigma,W,mean(f));
        end
    end
end
if length(XTest)==1
    RMSEtest=RMSEtest{1};
end