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

dff<-read.csv("clean_dff.csv")
set.seed(3749)


r1<-data.frame(dff[dff$SeriousDlqin2yrs==1,])
r0<-data.frame(dff[dff$SeriousDlqin2yrs==0,])

# ####################################################################################
# D1. create a cv-5 scheme with training data set
k<-5
train1_folds <- cut(seq(1,nrow(r1)),breaks=k,labels=FALSE)
train0_folds <- cut(seq(1,nrow(r0)),breaks=k,labels=FALSE)
train<-list()
train_rose<-list()
train_rs_tmk<-list()
lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")


for (i in 1:k){
  t<-data.frame(rbind(r1[which(train1_folds==i),],r0[which(train0_folds==i),]))
  train[[i]]<-t
  train_rose[[i]]<-ROSE(src~.,data=t)
  tomek<-TomekClassif(src~.,train_rose[[i]],Cl = "all", rem = "maj")
  train_rs_tmk[[i]]<-tomek[[1]]
  
  for (j in 1:k){
    task_orig<-makeClassifTask(data = train[[j]], target = "src",positive = "r2")
    train_orig<-train(lrn_rf,task_orig)
    predict_orig<-predict(train_orig,newdata = xft)
    
  }
}
remove(train0_folds,train1_folds)


