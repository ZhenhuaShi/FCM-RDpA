clc; clearvars; close all; rng(0);
nRepeats=8;
nRs=2.^(1:8); % number of rules
alphas=[1,.1,.01,.001,.0001]; % initial learning rate
lambda=0.05; % L2 regularization coefficient
Ps=0.5; % DropRule rate
gammaPs=0.5; % powerball param
nIt=1000; % number of iterations
Nbs=64; % batch size
GDs={'AdaBound','SGDM','Adam','AdaBelief'};
LN00={'RD-','RD-p'};
LN0=strcat(repmat(LN00,1,length(GDs)),reshape(repmat(GDs,length(LN00),1),1,[]));
LN0=strrep(LN0,'RD-AdaBound','RDA');
LN0=strrep(LN0,'RD-pAdaBelief','RDpA');
LN=cell(1,length(LN0)*length(nRs)+1);
LN(1)={'RR'};
for i=1:length(nRs)
    LN(2+(i-1)*length(LN0):1+i*length(LN0))=strcat(LN0, ['-nR' num2str(nRs(i))]);
end
nAlgs=length(LN);

datasets={'Concrete-CS';'Concrete-Flow';'Concrete-Slump';'tecator-fat';'tecator-moisture';'tecator-protein';'Yacht';'autoMPG';'NO2';'PM10';'Housing';'CPS';'EnergyEfficiency-Cooling';'EnergyEfficiency-Heating';'Concrete';'Airfoil';'Wine-red';'Abalone';'Abalone-onehot';'Wine-white';'PowerPlant';'Protein'};
datasets=datasets(1)

% Display results in parallel computing
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % print progress of parfor

[RMSEtrain,RMSEtest,RMSEtune]=deal(cellfun(@(u)nan(length(datasets),nAlgs,nIt),cell(nRepeats,1),'UniformOutput',false));
[times,BestP,Bestalpha,BestgammaP]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
BestmIter=cellfun(@(u)ones(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
thres=cellfun(@(u)inf(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
delete(gcp('nocreate'))
parpool(nRepeats);
parfor r=1:nRepeats
    dataDisp=cell(1,2);    dataDisp{1}=r;
    for s=1:length(datasets)
        dataDisp{2} = s;   send(dqWorker,dataDisp); % Display progress in parfor
        
        temp=load(['./' datasets{s} '.mat']);
        data=temp.data;
        X=data(:,1:end-1); y=data(:,end); y=y-mean(y);
        X = zscore(X); [N0,M]=size(X);
        N=round(N0*.7);
        
        idsTrain=datasample(1:N0,N,'replace',false);
        XTrain=X(idsTrain,:); yTrain=y(idsTrain);
        XTest=X; XTest(idsTrain,:)=[];
        yTest=y; yTest(idsTrain)=[];
        % validation data
        N1=round(N0*.15);
        idsTune=datasample(1:(N0-N),N1,'replace',false);
        XTune=XTest(idsTune,:); yTune=yTest(idsTune);
        XTest(idsTune,:)=[]; yTest(idsTune)=[];
        idsTest=1:N0;idsTest([idsTrain idsTune])=[];
        trainInd=idsTrain;
        testInd=1:N0;testInd(idsTrain)=[];
        valInd=testInd(idsTune);
        testInd(idsTune)=[];
        MXTrain=mean(XTrain);
        XTrain=XTrain-MXTrain; XTune=XTune-MXTrain; XTest=XTest-MXTrain;
        
        
        %% 1. Ridge regression
        tic
        id=1;
        b = ridge(yTrain,XTrain,lambda,0);
        RMSEtrain{r}(s,id,:) = sqrt(mean((yTrain-[ones(N,1) XTrain]*b).^2));
        RMSEtune{r}(s,id,:) = sqrt(mean((yTune-[ones(N1,1) XTune]*b).^2));
        RMSEtest{r}(s,id,:) = sqrt(mean((yTest-[ones(length(yTest),1) XTest]*b).^2));
        times{r}(s,id)=toc;
        
        for nRules=nRs
            
            %% Same init
            W0=zeros(nRules,M+1); % Rule consequents
            % FCM initialization
            [C0,U] = FuzzyCMeans(XTrain,nRules,[2 100 0.001 0]);
            Sigma0=C0;
            for ir=1:nRules
                Sigma0(ir,:)=std(XTrain,U(ir,:));
                W0(ir,1)=U(ir,:)*yTrain/sum(U(ir,:));
            end
            Sigma0(Sigma0==0)=mean(Sigma0(:));
            
            
            %% AdaBound
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=1
                        [tmp,tmpt]=FCM_RDpAdaBound(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=gammaPs
                        [tmp,tmpt]=FCM_RDpAdaBound(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),mIter]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            BestmIter{r}(s,id)=mIter;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            
            
            %% SGDM
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=1
                        [tmp,tmpt]=FCM_RDpSGDM(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=gammaPs
                        [tmp,tmpt]=FCM_RDpSGDM(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),mIter]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            BestmIter{r}(s,id)=mIter;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            
            
            %% Adam
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=1
                        [tmp,tmpt]=FCM_RDpAdam(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=gammaPs
                        [tmp,tmpt]=FCM_RDpAdam(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),mIter]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            BestmIter{r}(s,id)=mIter;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            
            
            %% AdaBelief
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=1
                        [tmp,tmpt]=FCM_RDpA(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),BestmIter{r}(s,id)]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            tic;
            id=id+1;
            for P=Ps
                for alpha=alphas
                    for gammaP=gammaPs
                        [tmp,tmpt]=FCM_RDpA(XTrain,yTrain,{XTune,XTest},{yTune,yTest},alpha,lambda,P,gammaP,nRules,nIt,Nbs,C0,Sigma0,W0);
                        if min(tmpt{1})<thres{r}(s,id)||~isfinite(thres{r}(s,id))
                            [thres{r}(s,id),mIter]=min(tmpt{1});
                            BestP{r}(s,id)=P;
                            Bestalpha{r}(s,id)=alpha;
                            BestgammaP{r}(s,id)=gammaP;
                            BestmIter{r}(s,id)=mIter;
                            [RMSEtrain{r}(s,id,:),RMSEtune{r}(s,id,:),RMSEtest{r}(s,id,:)]=deal(tmp,tmpt{1},tmpt{2});
                        end
                    end
                end
            end
            times{r}(s,id)=toc;
            
            
        end
    end
end
save('demoGD.mat','RMSEtrain','RMSEtune','RMSEtest','times','BestP','Bestalpha','BestmIter','BestgammaP','datasets','nAlgs','Nbs','LN','lambda','nRepeats','alphas','Ps','gammaPs','thres','LN0','nRs','nIt');

%% Plot results
ids=1:length(LN);
[tmp,ttmp]=deal(nan(length(datasets),length(LN),nRepeats));
for s=1:length(datasets)
    ttmp0=cellfun(@(u)squeeze(u(s,ids)),times,'UniformOutput',false);
    ttmp(s,ids,:)=cat(1,ttmp0{:})';
    for id=1:length(LN)
        tmp(s,id,:)=cell2mat(cellfun(@(u,m)squeeze(u(s,id,m(s,id))),RMSEtest,BestmIter,'UniformOutput',false));
    end
end
A=[nanmean(nanmean(tmp(:,ids,:),1),3);
    nanstd(nanmean(tmp(:,ids,:),1),[],3);
    nanmean(nanmean(ttmp(:,ids,:),1),3);
    nanstd(nanmean(ttmp(:,ids,:),1),[],3);
    nanmean(cat(1,Bestalpha{:}),1);
    nanmean(cat(1,BestP{:}),1);
    nanmean(cat(1,BestmIter{:}),1)
    nanmean(cat(1,thres{:}),1)];
a=squeeze(nanmean(tmp(:,ids,:),3));
a=[a;nanmean(a,1)]; sa=sort(a,2);
b=a==sa(:,1);c=a==sa(:,2);
at=squeeze(nanmean(ttmp(:,ids,:),3));
aa=nanmean(cat(3,Bestalpha{:}),3); aa=[aa;nanmean(aa,1)];
ap=nanmean(cat(3,BestP{:}),3); ap=[ap;nanmean(ap,1)];
am=nanmean(cat(3,BestmIter{:}),3); am=[am;nanmean(am,1)];

avgRMSE=nanmean(nanmean(tmp,1),3);
stdRMSE=nanstd(nanmean(tmp,1),[],3);
[iavgRMSE,istdRMSE,iavgTIME,istdTIME]=deal(nan(length(LN0)+1,length(nRs)));
iavgRMSE(1,:)=repmat(avgRMSE(:,1),1,length(nRs));
istdRMSE(1,:)=repmat(stdRMSE(:,1),1,length(nRs));
for i=2:length(LN0)+1
    iavgRMSE(i,:)=avgRMSE(:,i:length(LN0):end);
    istdRMSE(i,:)=stdRMSE(:,i:length(LN0):end);
end
avgTIME=nanmean(nanmean(ttmp,1),3);
stdTIME=nanstd(nanmean(ttmp,1),[],3);
iavgTIME(1,:)=repmat(avgTIME(:,1),1,length(nRs));
istdTIME(1,:)=repmat(stdTIME(:,1),1,length(nRs));
for i=2:length(LN0)+1
    iavgTIME(i,:)=avgTIME(:,i:length(LN0):end);
    istdTIME(i,:)=stdTIME(:,i:length(LN0):end);
end

close all
color={'k','g','b','r','m','c','#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
style={'-','--'};
lineStyles=cell(2,length(color)*length(style));
for i=1:length(color)
    for j=1:length(style)
        lineStyles{1,length(style)*(i-1)+j}=color{i};
        lineStyles{2,length(style)*(i-1)+j}=style{j};
    end
end

figure;
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',12);
idM=1:length(LN0);
f=mat2cell(permute(tmp(:,idM+(1:length(LN0):size(tmp,2)-length(LN0))',:),[3,2,1]),ones(1,nRepeats));
f=cellfun(@(x)permute(x,[2,3,1]),f,'UniformOutput',false);
RR=mat2cell(permute(tmp(:,1,:),[3,2,1]),ones(1,nRepeats));
RR=cellfun(@(x)permute(x,[2,3,1]),RR,'UniformOutput',false);
fR=cellfun(@(x,y)reshape(nanmean(x./y,2),length(nRs),[])',f,RR,'UniformOutput',false);
fR=cat(3,fR{:});
savgRMSE=nanmean(fR,3);
sstdRMSE=nanstd(fR,[],3);
for i=1:length(idM)
    errorbar(nRs, savgRMSE(i,:),sstdRMSE(i,:),'Color',lineStyles{1,i},'LineStyle',lineStyles{2,i},'linewidth',2);
    hold on;
end
set(gca,'XTick',nRs);
xlabel('$R$','interpreter','latex');
ylabel('Average normalized test RMSE');
set(gca,'yscale','log','xscale','log');
box on; axis tight;
legend(LN0(idM),'NumColumns',1,'Location','eastoutside');

figure;
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',12);
for i=1:length(idM)
    errorbar(nRs, iavgTIME(idM(i)+1,:),istdTIME(idM(i)+1,:),'Color',lineStyles{1,i},'LineStyle',lineStyles{2,i},'linewidth',2);
    hold on;
end
set(gca,'XTick',nRs);
xlabel('$R$','interpreter','latex');
ylabel('Time (s)');
set(gca,'yscale','log','xscale','log');
box on; axis tight;
legend(LN0(idM),'NumColumns',1,'Location','eastoutside');