function [RMSEtrain,RMSEtest,CB,SigmaB,WB,fB]=PCA_GP_RDpA(XTrain,yTrain,XTest,yTest,alpha,rr,P,gammaP,nMFs,nIt,Nbs,C0,Sigma0,W0)
% %% Inputs:
% XTrain: N*M matrix of the training inputs. N is the number of samples, and M the feature dimensionality.
% yTrain: N*1 vector of the labels for XTrain
% XTest: {NValidation*M, NTest*M} matrix cell of the validation and test inputs 
% yTest: {NValidation*1, NTest*1} vector cell of the labels for XTest
% alpha: scalar, initial learning rate
% rr: scalar, L2 regularization coefficient 
% P: scalar, DropRule preservation rate
% gammaP: scalar, Powerball power exponent
% nMFs: scalar, number of MFs in each input domain
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
if Nbs>N; Nbs=N; end
nRules=nMFs^M; % number of rules
if nargin<12
    C0=zeros(M,nMFs); Sigma0=C0; W0=zeros(nRules,M+1);
    for m=1:M % Initialization
        C0(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),nMFs);
        Sigma0(m,:)=std(XTrain(:,m));
    end
end
C=C0; Sigma=Sigma0; W=W0;
minSigma=min(Sigma(:));

[CB,SigmaB,WB,fB]=deal(C,Sigma,W,zeros(1,nRules));
%% Iterative update
RMSEtrain=zeros(1,nIt); RMSEtest=cellfun(@(u)RMSEtrain,XTest,'UniformOutput',false);
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; yPred=nan(Nbs,1);
for it=1:nIt
    deltaC=zeros(M,nMFs); deltaSigma=deltaC;  deltaW=rr*W; deltaW(:,1)=0; % consequent
    f=zeros(Nbs,nRules); % firing level of rules
    idsTrain=datasample(1:N,Nbs,'replace',false);
    idsGoodTrain=true(Nbs,1);
    for n=1:Nbs
        mu=exp(-(XTrain(idsTrain(n),:)'-C).^2./(2*Sigma.^2));
        deltamuC=(XTrain(idsTrain(n),:)'-C)./(Sigma.^2);
        deltamuSigma=(XTrain(idsTrain(n),:)'-C).^2./(Sigma.^3);
        for m=1:M % membership grades of MFs
            if m==1
                pmu=mu(m,:);
                [deltapmuC,deltapmuSigma]=deal(zeros(1,nMFs,nMFs));
                deltapmuC(1,:,:)=diag(deltamuC(m,:));
                deltapmuSigma(1,:,:)=diag(deltamuSigma(m,:));
            else
                pmu=[repmat(pmu,1,nMFs); reshape(repmat(mu(m,:),size(pmu,2),1),1,[])];
                deltapmuC=[repmat(deltapmuC,1,nMFs); permute(reshape(repmat(diag(deltamuC(m,:)),size(deltapmuC,2),1),nMFs,[]),[3,2,1])];
                deltapmuSigma=[repmat(deltapmuSigma,1,nMFs); permute(reshape(repmat(diag(deltamuSigma(m,:)),size(deltapmuSigma,2),1),nMFs,[]),[3,2,1])];
            end
        end
        idsKeep=rand(1,nRules)<=P;
        f(n,idsKeep)=prod(pmu(:,idsKeep),1);
        if sum(~isfinite(f(n,idsKeep)))
            continue;
        end
        if ~sum(f(n,idsKeep)) % special case: all f(n,:)=0; no dropRule
            idsKeep=~idsKeep;
            f(n,idsKeep)=prod(pmu(:,idsKeep),1);
            idsKeep=true(1,nRules);
        end
        deltapmuC=deltapmuC(:,idsKeep,:);
        deltapmuSigma=deltapmuSigma(:,idsKeep,:);
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
            deltaC=deltaC+permute(sum(deltaYmu.*deltapmuC,2),[1,3,2]);
            deltaSigma=deltaSigma+permute(sum(deltaYmu.*deltapmuSigma,2),[1,3,2]);
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
    
    % Training RMSE
    RMSEtrain(it)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
    % Test RMSE
    for i=1:length(XTest)
        NTest=size(XTest{i},1);
        f=zeros(NTest,nRules); % firing level of rules
        for n=1:NTest
            mu=exp(-(XTest{i}(n,:)'-C).^2./(2*Sigma.^2));
            for m=1:M % membership grades of MFs
                if m==1
                    pmu=mu(m,:);
                else
                    pmu=[repmat(pmu,1,nMFs); reshape(repmat(mu(m,:),size(pmu,2),1),1,[])];
                end
            end
            f(n,:)=prod(pmu,1);
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
end