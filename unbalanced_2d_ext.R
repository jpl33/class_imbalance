require(MASS)
require(ggplot2)
require(corpcor)
require(FNN)
require(randomForest)
require(DMwR)
require(ROSE)
require(unbalanced)
require(UBL)
require(mlr)
require(gridExtra)


set.seed(2385)
# A. build arbitrary covariance matrices 
#X1 <- matrix(seq(1,16,by=4), 2, 2)
X1 <- matrix(c(4, 1.2, 2.5, 4), 2)
X1<-make.positive.definite(X1)

#X2<-matrix(seq(12,1,-3),2,2)
X2<-matrix(c(7, 3, 3, 8), 2)
# oops! watch it, covariance matrix MUST be positive definite matrix 
X2<-make.positive.definite(X2)

# # B. create 2 2-d noramlly dostributed data sets according to covariance matrices and arbitrary means: not too close, not too far
r1<-mvrnorm(2000,c(10,8),as.matrix(X1))
r2<-mvrnorm(200,c(6,7),as.matrix(X2))
r1<-cbind(r1,rep(as.numeric(0), nrow(r1)))
r2<-cbind(r2,rep(as.numeric(1), nrow(r2)))
# combine the two data sets into a single matrix 
XX<-rbind(r1,r2)
XX2<-XX[sample(seq(1,nrow(XX)),nrow(XX)),]

# # ggplot is sensitive to the data format:so let's turn XX (matrix) into a data.frame
xfd<-data.frame(X1=XX2[,1], X2=XX2[,2],src=as.factor(XX2[,3]))
levels(xfd$src)<-c("r1","r2")# we are renaming the test response variable to match the train values so we can compare Test Vs.Train sets

ggplot(xfd, aes(x=xfd$X1,y=xfd$X2,colour=src,shape=21))+geom_point(aes(size=5))+geom_text(aes(label=rownames(xfd)), size=4)+scale_shape_identity()

r3<-mvrnorm(2000,c(10,8),as.matrix(X1))
r4<-mvrnorm(200,c(6,7),as.matrix(X2))
r3<-cbind(r3,rep(as.numeric(0), nrow(r3)))
r4<-cbind(r4,rep(as.numeric(1), nrow(r4)))
# combine the two data sets into a single matrix 
XT<-rbind(r3,r4)

xft<-data.frame(X1=XT[,1],X2=XT[,2],src=as.factor(XT[,3]))
levels(xft$src)<-c("r1","r2")# we are renaming the test response variable to match the train values so we can compare Test Vs.Train sets


orig_plot<-ggplot(xfd, aes(x=xfd$X1,y=xfd$X2,colour=src))+geom_point(aes(size=4,shape=21))+geom_text(aes(label=rownames(xfd)))+scale_shape_identity()+ggtitle("orig_plot")

task_orig<-makeClassifTask(data = xfd, target = "src",positive = "r2")
lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")
train_orig<-train(lrn_rf,task_orig)
predict_orig<-predict(train_orig,newdata = xft)

ub_y<-as.factor(xfd$src)
levels(ub_y)<-c(0,1)
t1<-10
xfd_ub_smt<-list()
rnames<-list()
xfd_ub_smt_ggplt<-list()
predict_smt<-list()
for (n in seq(1,t1,by=2)){
  fdt<-ubSMOTE(xfd[,1:2],ub_y, perc.over = n*100, k = 5, perc.under = 100, verbose = TRUE)
  xfd_ub_smt[[n]]<-data.frame(fdt[[1]],"src"=fdt[[2]])
  levels(xfd_ub_smt[[n]]$src)<-c("r1","r2")
  dd<-xfd_ub_smt[[n]]
  dd[,"rnames"]<-rownames(xfd_ub_smt[[n]])
  xfd_ub_smt_ggplt[[n]]<-ggplot(dd, aes(x=X1,y=X2,colour=src))+geom_point(aes(size=4, shape = 21))+geom_text(aes(label=rnames))+scale_shape_identity()+ggtitle(paste0("k=",n))
  task_smt<-makeClassifTask(data = xfd_ub_smt[[n]], target = "src",positive = "r2")
  lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")
  train_smt<-train(lrn_rf,task_smt)
  predict_smt[[n]]<-predict(train_smt,newdata = xft)
}

grid.arrange(orig_plot,xfd_ub_smt_ggplt[[1]],xfd_ub_smt_ggplt[[3]],xfd_ub_smt_ggplt[[5]],xfd_ub_smt_ggplt[[7]], ncol=2)
grid_roc<-generateThreshVsPerfData(list(orig=predict_orig,k_1 = predict_smt[[1]],k_3 = predict_smt[[3]],k_5 = predict_smt[[5]],k_7 = predict_smt[[7]]), measures =  list(fpr, tpr,auc,gmean))
 ggplot(grid_roc$data,aes(x=grid_roc$data$fpr, y=grid_roc$data$tpr,colour=grid_roc$data$learner))+geom_line()
 grid_roc$data$learner[which(grid_roc$data$gmean==max(grid_roc$data$gmean))]
 grid_roc$data$learner[which(grid_roc$data$auc==max(grid_roc$data$auc))]
  
 

# 
# xfd_rose<-ROSE(src~.,data=xfd,N=1.5*nrow(xfd))
# xfd_rose<-data.frame(xfd_rose$data[,1:2],"src"=as.factor(xfd_rose$data[,3]))
# ggplot(xfd_rose, aes(x=X1,y=X2,colour=src))+geom_point(aes(size=4, shape = 21))+geom_text(aes(label=rownames(xfd_rose)))+scale_shape_identity()
# 
# 
# trf_rose<-  randomForest(x=xfd_rose[,1:2], y=xfd_rose[,3],mtry=1)
# preds.trf_rose<- predict(trf_rose)
# 
# task_rose<-makeClassifTask(data = xfd_rose, target = "src",positive = "r2")
# train_rose<-train(lrn_rf,task_rose)
# predict_rose<-predict(train_rose,newdata = xft)
# roc_rose<-generateThreshVsPerfData(predict_rose,measures =  list(fpr, tpr,gmean,auc))
# roc<-generateThreshVsPerfData(list(rose = predict_rose,SMOTE= predict_smt), measures =  list(fpr, tpr,auc,gmean))
# ggplot(roc$data,aes(x=roc$data$fpr, y=roc$data$tpr,colour=roc$data$learner))+geom_line()
# 
# xfd_tomek<-xfd
# xfd_tomek$src<-as.factor(xfd_tomek$src)
# #xfd_tomek2<-xfd
# tomek<-TomekClassif(src~.,xfd_tomek)
# #tomek2<-ubTomek(xfd_tomek2[,1:2],ub_y)
# xfd_tomek[,"tomek"]<-0
# xfd_tomek[rownames(xfd_tomek) %in% tomek[[2]],"tomek"]<-1
# ggplot(xfd_tomek,aes(x=X1, y=X2,colour=src,fill = as.factor(tomek)))+geom_point(aes(size=3,shape=21))+scale_shape_identity()+scale_fill_manual(values=c(NA,"green"))+geom_text(aes(label=rownames(xfd_tomek)))
# 
# # xfd_tomek2[,"tomek"]<-0
# # xfd_tomek2[rownames(xfd_tomek) %in% tomek2$id.rm,"tomek"]<-1
# # ggplot(xfd_tomek2,aes(x=X1, y=X2,colour=src,fill = as.factor(tomek)))+geom_point(aes(size=3,shape=21))+scale_shape_identity()+scale_fill_manual(values=c(NA,"green"))+geom_text(aes(label=rownames(xfd_tomek2)))
# 
# 
# 
# #knn_qry<-xfd_tomek[rownames(xfd_tomek) %in% tomek$id.rm,1:2]
# t<-4
# z<-xfd_tomek[tomek[[2]] ,]
# tomek_min<-as.numeric(row.names( z[z$src=="r2",]))
# knn_qry<-xfd_tomek[tomek_min,]
# tomek_knn<-get.knnx(xfd_tomek[,1:2],knn_qry[,1:2],k=t)
# 
# tomek_knn$nn.index<-cbind( tomek_knn$nn.index,"rm"=as.numeric(0))
# for (i in 1:nrow(tomek_knn$nn.index)){
#   rm_id<-0
#   for (j in 2:t){
#   rm_id<-ifelse(xfd_tomek[tomek_knn$nn.index[i,1],"src"]==xfd_tomek[tomek_knn$nn.index[i,j],"src"],rm_id<-rm_id+1,rm_id)
#   }
#   if (rm_id<(t/2)){tomek_knn$nn.index[i,"rm"]<-1}
# }
# 
