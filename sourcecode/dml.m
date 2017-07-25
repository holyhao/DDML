
dst_dim=10; % dimensionality of the target latent space
lambda=0; %tradeoff parameter of the regularizer
num_epoch=1; % number of training epoches
lr=1;  %learning rate
mb_size=100; %minibatch size
freq_eval=5000; %frequency to evaluate the objective function


X=load('train_fea.mat');
X=X.train_fea;
simi_pairs=load('train_simi_pairs.txt');
diff_pairs=load('train_diff_pairs.txt');
num_simi_pairs=size(simi_pairs,1);
src_dim=size(X,2);

L=rand(dst_dim, src_dim)*0.01; % L should be initialized such that the distance between each data pair is around 1.
tic;

update=0;
for e=1:num_epoch
    
    accu_grad=zeros(size(L));
    for d=1:size(simi_pairs,1)
        if(mod(d,freq_eval)==0||d==1)
            %evaluate
            time=toc;
            simi_loss=0;
            diff_loss=0;
            cnt=0;
            simicnt=0;diffcnt=0;
            for p=1:100:size(simi_pairs,1)
                x=simi_pairs(p,1);
                y=simi_pairs(p,2);
                diff=X(x,:)-X(y,:);
                dis=L*diff';
                dis=sum(dis.^2);
                simi_loss=simi_loss+dis;
                simicnt=simicnt+1;
                x=diff_pairs(p,1);
                y=diff_pairs(p,2);
                diff=X(x,:)-X(y,:);
                dis=L*diff';
                dis=sum(dis.^2);
                if(dis<1)
                    diff_loss=diff_loss+(1-dis);
                end
                diffcnt=diffcnt+1;
            end
            loss=simi_loss/simicnt+diff_loss/diffcnt;
            if lambda~=0
                loss=loss-lambda*logdet(L*L');
            end
            disp(['update: ' num2str(update) ' loss: ' num2str(loss) ...
                ' simi loss: ' num2str(simi_loss/simicnt) ...
                ' diff loss: ' num2str(diff_loss/diffcnt) ...
                ' time: ' num2str(time)]);
            tic;
        end
        
        %update with simi_pair
        x=simi_pairs(d,1);
        y=simi_pairs(d,2);
        diff=X(x,:)-X(y,:);
        grad=2*(L*diff')*diff;
        accu_grad=accu_grad+grad;
        %update with diff_pair
        x=diff_pairs(d,1);
        y=diff_pairs(d,2);
        diff=X(x,:)-X(y,:);
        dis=L*diff';
        dis=sum(dis.^2);
        if(dis<1)
            grad=-2*(L*diff')*diff;
            accu_grad=accu_grad+grad;
        end
        if(mod(d,mb_size)==0)
            if lambda~=0
                L=L-lr*(accu_grad/mb_size/2-lambda*inv(L*L')*L);
            else
                L=L-lr*(accu_grad/mb_size/2);
            end
            update=update+1;
            accu_grad=zeros(size(L));
        end
    end
end
dlmwrite(['L_' num2str(lambda) '_' num2str(dst_dim) '.txt'], L, ' ');
