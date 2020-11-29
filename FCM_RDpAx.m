function [RMSEtrain,RMSEtest,WAB,WCB,CB,SigmaB,WB,fB]=FCM_RDpAx(XTrain,yTrain,XTest,yTest,nFeatures,alpha,rr,P,gammaP,nRules,nIt,Nbs,WA0,WC0,C0,Sigma0,W0)
% This function implements the FCM_RDpAx algorithm in the following paper:
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
% WA0: M*nFeatures initialization matrix of the linear projection for the Antecedent
% WC0: M*nFeatures initialization matrix of the linear projection for the Consequent
% C0: nRules*(M+nFeatures) initialization matrix of the centers of the Gaussian MFs
% Sigma0: nRules*(M+nFeatures) initialization matrix of the standard deviations of the Gaussian MFs
% W0: nRules*(M+nFeatures+1) initialization matrix of the consequent parameters for the nRules rules
%
% %% Outputs:
% RMSEtrain: 1*nIt vector of the training RMSE at different iterations
% RMSEtest: {1*nIt, 1*nIt} vector cell of the validation and test RMSE at different iterations
% WAB: M*nFeatures matrix of the linear projection for the Antecedent
% WCB: M*nFeatures matrix of the linear projection for the Consequent
% CB: nRules*(M+nFeatures) matrix of the centers of the Gaussian MFs
% SigmaB: nRules*(M+nFeatures) matrix of the standard deviations of the Gaussian MFs
% WB: nRules*(M+nFeatures+1) matrix of the consequent parameters for the nRules rules
% fB: 1*nRules vector of the firing levels for validation inputs

beta1=0.9; beta2=0.999; thre=inf;

if ~iscell(XTest)
    XTest={XTest};
    yTest={yTest};
end
[N,~]=size(XTrain);
Nbs=min(N,Nbs);
if nargin<13
    % PCA, assume XTrain is centered.
    WA0=pca(XTrain,'NumComponents',nFeatures);
    WA0=[zeros(1,size(WA0,2)); WA0];
    WC0=WA0;
    XTrainA=[ones(size(XTrain,1),1) XTrain]*WA0;  
    XTrainB=[XTrainA XTrain];
    W0=zeros(nRules,size(XTrainB,2)+1); % Rule consequents
    % FCM initialization
    [C0,U] = fcm(XTrainB,nRules,[2 100 0.001 0]);
    Sigma0=C0;
    for r=1:nRules
        Sigma0(r,:)=std(XTrainB,U(r,:));
        W0(r,1)=U(r,:)*yTrain/sum(U(r,:));
    end
    Sigma0(Sigma0==0)=mean(Sigma0(:));
end
WA=WA0; WC=WC0; C=C0; Sigma=Sigma0; W=W0;
minSigma=.1*min(Sigma0(:));
    
[WAB,WCB,CB,SigmaB,WB,fB]=deal(WA,WC,C,Sigma,W,zeros(1,nRules));
%% Iterative update
RMSEtrain=zeros(1,nIt); RMSEtest=cellfun(@(u)RMSEtrain,XTest,'UniformOutput',false);
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
[mWA,vWA,mWC,vWC]=deal(0);
for it=1:nIt
    deltaC=zeros(nRules,size(XTrain,2)+nFeatures); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    deltaXA=zeros(Nbs,nFeatures);
    deltaXC=zeros(Nbs,nFeatures);
    f=zeros(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    % DRX
    XTrainA=[ones(Nbs,1) XTrain(idsTrain,:)]*WA;
    XTrainB=[XTrainA XTrain(idsTrain,:)];
    XTrainC=[ones(Nbs,1) XTrain(idsTrain,:)]*WC;
    XTrainD=[XTrainC XTrain(idsTrain,:)];
    for n=1:Nbs
        idsKeep=rand(1,nRules)<=P;
        f(n,idsKeep)=prod(exp(-(XTrainB(n,:)-C(idsKeep,:)).^2./(2*Sigma(idsKeep,:).^2)),2);
        if sum(~isfinite(f(n,idsKeep)))
            continue;
        end
        if ~sum(f(n,idsKeep)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=prod(exp(-(XTrainB(n,:)-C(idsKeep,:)).^2./(2*Sigma(idsKeep,:).^2)),2);
            idsKeep=true(1,nRules);
        end
        deltamuC=(XTrainB(n,:)-C(idsKeep,:))./(Sigma(idsKeep,:).^2);
        deltamuSigma=(XTrainB(n,:)-C(idsKeep,:)).^2./(Sigma(idsKeep,:).^3);
        fBar=f(n,idsKeep)/sum(f(n,idsKeep));
        yR=[1 XTrainD(n,:)]*W(idsKeep,:)';
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
            deltaXA(n,:)=deltaXA(n,:)-sum(deltaYmu'.*deltamuC(:,1:nFeatures),1);
            deltaSigma(idsKeep,:)=deltaSigma(idsKeep,:)+deltaYmu'.*deltamuSigma;
            deltaW(idsKeep,:)=deltaW(idsKeep,:)+(yPred(n)-yTrain(idsTrain(n)))*fBar'*[1 XTrainD(n,:)];
            deltaXC(n,:)=deltaXC(n,:)+(yPred(n)-yTrain(idsTrain(n)))*fBar*W(idsKeep,2:1+nFeatures);
        end
    end
    deltaWA=[ones(Nbs,1) XTrain(idsTrain,:)]'*deltaXA;
    deltaWC=[ones(Nbs,1) XTrain(idsTrain,:)]'*deltaXC;
    
    % powerball
    deltaC=sign(deltaC).*(abs(deltaC).^gammaP);
    deltaSigma=sign(deltaSigma).*(abs(deltaSigma).^gammaP);
    deltaW=sign(deltaW).*(abs(deltaW).^gammaP);
    deltaWA=sign(deltaWA).*(abs(deltaWA).^gammaP);
    deltaWC=sign(deltaWC).*(abs(deltaWC).^gammaP);
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
    
    mWA=beta1*mWA+(1-beta1)*deltaWA;
    vWA=beta2*vWA+(1-beta2)*(deltaWA-mWA).^2;
    mWAHat=mWA/(1-beta1^it);
    vWAHat=vWA/(1-beta2^it);
    WA=WA-alpha*mWAHat./(sqrt(vWAHat)+10^(-8));
    
    mWC=beta1*mWC+(1-beta1)*deltaWC;
    vWC=beta2*vWC+(1-beta2)*(deltaWC-mWC).^2;
    mWCHat=mWC/(1-beta1^it);
    vWCHat=vWC/(1-beta2^it);
    WC=WC-alpha*mWCHat./(sqrt(vWCHat)+10^(-8));
    
    % Training RMSE on the minibatch
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    for i=1:length(XTest)
        NTest=size(XTest{i},1);
        XTestA=[ones(NTest,1) XTest{i}]*WA;
        XTestB=[XTestA XTest{i}];
        XTestC=[ones(NTest,1) XTest{i}]*WC;
        XTestD=[XTestC XTest{i}];
        f=zeros(NTest,nRules); % firing level of rules
        for n=1:NTest
            f(n,:)=prod(exp(-(XTestB(n,:)-C).^2./(2*Sigma.^2)),2);
        end
        f(:,P==0)=0;
        yR=[ones(NTest,1) XTestD]*W';
        yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
        yPredTest(isnan(yPredTest))=nanmean(yPredTest);
        RMSEtest{i}(it)=sqrt((yTest{i}-yPredTest)'*(yTest{i}-yPredTest)/NTest);
        if isnan(RMSEtest{i}(it)) && it>1
            RMSEtest{i}(it)=RMSEtest{i}(it-1);
        end
        if nargout>2&&i==1&&RMSEtest{i}(it)<thre
            thre=RMSEtest{i}(it);
            [WAB,WCB,CB,SigmaB,WB,fB]=deal(WA,WC,C,Sigma,W,mean(f));
        end
    end
end
if length(XTest)==1
    RMSEtest=RMSEtest{1};
end
