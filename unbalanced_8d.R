require(MASS)
require(ggplot2)
require(corpcor)
require(FNN)
require(randomForest)
require(DMwR)
require(ROSE)
require(UBL)
require(mlr)
require(clusterGeneration)

set.seed(2385)
# A. build arbitrary covariance matrices 
# oops! watch it, covariance matrix MUST be positive definite matrix 


X1<- genPositiveDefMat(8,lambdaLow=3,ratioLambda = 6)
X2<- genPositiveDefMat(8,lambdaLow=3,ratioLambda = 5)

# # B. create 2 2-d noramlly dostributed data sets according to covariance matrices and arbitrary means: not too close, not too far
r1<-mvrnorm(200,rnorm(8, mean = 8, sd = 1.5),as.matrix(X1$Sigma))
r2<-mvrnorm(20,rnorm(8, mean = 6, sd = 1.5),as.matrix(X2$Sigma))
r1<-cbind(r1,rep(as.numeric(0), nrow(r1)))
r2<-cbind(r2,rep(as.numeric(1), nrow(r2)))
# combine the two data sets into a single matrix 
XX<-rbind(r1,r2)
XX<-XX[sample(seq(1,nrow(XX)),nrow(XX)),]

# # ggplot is sensitive to the data format:so let's turn XX (matrix) into a data.frame
xfd<-data.frame(XX[,1:(ncol(XX)-1)],src=as.factor(XX[,ncol(XX)]))
levels(xfd$src)<-c("r1","r2")# we are renaming the test response variable to match the train values so we can compare Test Vs.Train sets

ggplot(xfd, aes(x=xfd$X1,y=xfd$X2,colour=src,shape=21))+geom_point(aes(size=5))+geom_text(aes(label=rownames(xfd)), size=4)+scale_shape_identity()

r3<-mvrnorm(200,rnorm(8, mean = 8, sd = 1.5),as.matrix(X1$Sigma))
r4<-mvrnorm(20,rnorm(8, mean = 6, sd = 1.5),as.matrix(X2$Sigma))
r3<-cbind(r3,rep(as.numeric(0), nrow(r3)))
r4<-cbind(r4,rep(as.numeric(1), nrow(r4)))
# combine the two data sets into a single matrix 
XT<-rbind(r3,r4)

xft<-data.frame(XT[,1:(ncol(XT)-1)],src=as.factor(XT[,ncol(XT)]))
levels(xft$src)<-c("r1","r2")# we are renaming the test response variable to match the train values so we can compare Test Vs.Train sets


task_orig<-makeClassifTask(data = xfd, target = "src",positive = "r2")
lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")
train_orig<-train(lrn_rf,task_orig)
predict_orig<-predict(train_orig,newdata = xft)

ub_y<-as.factor(xfd$src)
levels(ub_y)<-c(0,1)
xfd_ub_smt<-ubSMOTE(xfd[,-ncol(xfd)],ub_y, perc.over = 0.5*100*(nrow(r1)/nrow(r2)), k = 5, perc.under = 100, verbose = TRUE)
xfd_smt<-SmoteClassif(src~.,xfd)
xfd_ub_smt<-data.frame(xfd_ub_smt[[1]],"src"=xfd_ub_smt[[2]])
levels(xfd_ub_smt$src)<-c("r1","r2")
row.names(xfd_ub_smt)<-seq(1,nrow(xfd_ub_smt))
ggplot(xfd_ub_smt, aes(x=X1,y=X2,colour=src))+geom_point(aes(size=4, shape = 21))+geom_text(aes(label=rownames(xfd_ub_smt)))+scale_shape_identity()

task_smt<-makeClassifTask(data = xfd_ub_smt, target = "src",positive = "r2")
lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")
train_smt<-train(lrn_rf,task_smt)
predict_smt<-predict(train_smt,newdata = xft)
roc_smt<-generateThreshVsPerfData(predict_smt,measures =  list(fpr, tpr,gmean))


xfd_rose<-ROSE(src~.,data=xfd)
xfd_rose<-data.frame(xfd_rose$data[,-ncol(xfd)],"src"=as.factor(xfd_rose$data[,ncol(xfd)]))
ggplot(xfd_rose, aes(x=X1,y=X2,colour=src))+geom_point(aes(size=4, shape = 21))+geom_text(aes(label=rownames(xfd_rose)))+scale_shape_identity()

task_rose<-makeClassifTask(data = xfd_rose, target = "src",positive = "r2")
train_rose<-train(lrn_rf,task_rose)
predict_rose<-predict(train_rose,newdata = xft)
roc_rose<-generateThreshVsPerfData(predict_rose,measures =  list(fpr, tpr,gmean,auc))

xfd_tomek<-xfd
tomek<-TomekClassif(src~.,xfd_tomek,Cl="all", rem = "maj")
xfd_tomek[,"tomek"]<-0
xfd_tomek[rownames(xfd_tomek) %in% tomek[[2]],"tomek"]<-1
ggplot(xfd_tomek,aes(x=X1, y=X2,colour=src,fill = as.factor(tomek)))+geom_point(aes(size=3,shape=21))+scale_shape_identity()+scale_fill_manual(values=c(NA,"green"))+geom_text(aes(label=rownames(xfd_tomek)))
xfd_tomek<-xfd_tomek[,1:ncol(xfd)]

# t<-4
# # find all Tomek link points
# z<-xfd_tomek[tomek[[2]] ,]
# # get the tomek link points from the minority class
# tomek_min<-as.numeric(row.names( z[z$src=="r2",]))
# knn_qry<-xfd_tomek[tomek_min,]
# # get the t nearest neighbors for the minority tomek link points
# tomek_knn<-get.knnx(xfd_tomek[,1:2],knn_qry[,1:2],k=t)
# 
# tomek_knn$nn.index<-cbind( tomek_knn$nn.index,"rm"=as.numeric(0))
# for (i in 1:nrow(tomek_knn$nn.index)){
#   rm_id<-0
#   for (j in 2:t){
#     # for each point, look up if its t nearest neighbors are  minority or majority
#     # if it's a minority- raise the index, if it's a majority - leave index at "0"
#   rm_id<-ifelse(xfd_tomek[tomek_knn$nn.index[i,1],"src"]==xfd_tomek[tomek_knn$nn.index[i,j],"src"],rm_id<-rm_id+1,rm_id)
#   }
#   # if rm_id is less than 2, most neighbors around the minority points are majority!
#   if (rm_id<(t/2)){tomek_knn$nn.index[i,"rm"]<-1}
# }
# # delete from xfd all tomek link minority members with more than half
# # their nearest neighbors belonging to the majority
# 
# if (!is.null(nrow(tomek_knn$nn.index[tomek_knn$nn.index[,t+1]==1,]))){
# xfd_tomek<-xfd_tomek[-(tomek_knn$nn.index[tomek_knn$nn.index[,t+1]==1,1]),]
# }
# 
# if (! is.null(nrow(tomek_knn$nn.index[tomek_knn$nn.index[,t+1]==0,]))){
#   tomek_maj<-xfd_tomek[rownames(xfd_tomek) %in% (tomek_knn$nn.index[tomek_knn$nn.index[,t+1]==0,2]),]
#   xfd_tomek<-xfd_tomek[-as.numeric(row.names(tomek_maj)),]
# }


task_tomek<-makeClassifTask(data = xfd_tomek, target = "src",positive = "r2")
train_tomek<-train(lrn_rf,task_tomek)
predict_tomek<-predict(train_tomek,newdata = xft)

# adj_tomek2<-function(df){
#   tomek_smt<-TomekClassif(src~.,xfd_ub_smt)
#   # find all Tomek link points
#   zsmt<-xfd_ub_smt[tomek_smt[[2]] ,]
#   # get the tomek link points from the minority class
#   tomek_min_smt<-as.numeric(row.names( zsmt[zsmt$src=="r2",]))
#   knn_qry_smt<-xfd_ub_smt[tomek_min_smt,]
#   # get the t nearest neighbors for the minority tomek link points
#   tomek_knn_smt<-get.knnx(xfd_ub_smt[,1:2],knn_qry_smt[,1:2],k=t)
#   tomek_knn_smt$nn.index<-cbind( tomek_knn_smt$nn.index,"rm"=as.numeric(0))
#   for (i in 1:nrow(tomek_knn_smt$nn.index)){
#     rm_id<-0
#     for (j in 2:t){
#       # for each point, look up if its t nearest neighbors are  minority or majority
#       # if it's a minority- raise the index, if it's a majority - leave index at "0"
#       rm_id<-ifelse(xfd_ub_smt[tomek_knn_smt$nn.index[i,1],"src"]==xfd_ub_smt[tomek_knn_smt$nn.index[i,j],"src"],rm_id<-rm_id+1,rm_id)
#     }
#     # if rm_id is less than 2, most neighbors around the minority points are majority!
#     if (rm_id<(t/2)){tomek_knn_smt$nn.index[i,"rm"]<-1}
#   }
#   # delete from xfd all tomek link minority members with more than half
#   # their nearest neighbors belonging to the majority
#   if (nrow(tomek_knn_smt$nn.index[tomek_knn_smt$nn.index[,t+1]==1,])>0){
#      xfd_ub_smt<-xfd_ub_smt[-(tomek_knn_smt$nn.index[tomek_knn_smt$nn.index[,t+1]==1,1]),]
#   }
#   
#   if (nrow(tomek_knn_smt$nn.index[tomek_knn_smt$nn.index[,t+1]==0,])>0){
#     tomek_maj<-xfd_ub_smt[rownames(xfd_ub_smt) %in% (tomek_knn_smt$nn.index[tomek_knn_smt$nn.index[,t+1]==0,2]),]
#     xfd_ub_smt<-xfd_ub_smt[-as.numeric(row.names(tomek_maj)),]
#   }
# }

tomek_smt<-TomekClassif(src~.,xfd_ub_smt)
xfd_smt_tomek<-tomek_smt[[1]]
task_smt_tomek<-makeClassifTask(data = xfd_smt_tomek, target = "src",positive = "r2")
train_smt_tomek<-train(lrn_rf,task_smt_tomek)
predict_smt_tomek<-predict(train_smt_tomek,newdata = xft)

tomek_rose<-TomekClassif(src~.,xfd_rose,Cl = "all", rem = "maj")
xfd_rose2<-tomek_rose[[1]]
rose_tomek<-ggplot(xfd_rose2, aes(x=X1,y=X2,colour=src))+geom_point(aes(size=4, shape = 21))+geom_text(aes(label=rownames(xfd_rose2)))+scale_shape_identity()+ggtitle("rose_tomek")

task_rose_tomek<-makeClassifTask(data = xfd_rose2, target = "src",positive = "r2")
train_rose_tomek<-train(lrn_rf,task_rose_tomek)
predict_rose_tomek<-predict(train_rose_tomek,newdata = xft)


roc<-generateThreshVsPerfData(list(orig=predict_orig,rose = predict_rose,SMOTE= predict_smt, tomek=predict_tomek,smt_tomek=predict_smt_tomek, rose_tomek=predict_rose_tomek), measures =  list(fpr, tpr,auc,gmean))
ggplot(roc$data,aes(x=roc$data$fpr, y=roc$data$tpr,colour=roc$data$learner))+geom_line()
library(dplyr)
roc$data %>% group_by(learner) %>%
  summarise(mval=max(auc), mgmean=max(gmean)) -> result