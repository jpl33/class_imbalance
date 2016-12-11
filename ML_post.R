require(MASS)
require(tmvtnorm)
require(randomForest)
require(nnet)
require(NeuralNetTools)
require(ggplot2)
require(ROCR)
require(DMwR)
dff<-read.csv("clean_dff.csv")

r1<-data.frame(dff[dff$SeriousDlqin2yrs==1,])
r0<-data.frame(dff[dff$SeriousDlqin2yrs==0,])


# ####################################################################################
# D1. create a cv-5 scheme with training data set
 
  train1_folds <- cut(seq(1,nrow(r1)),breaks=5,labels=FALSE)
  train0_folds <- cut(seq(1,nrow(r0)),breaks=5,labels=FALSE)
  train<-list()
  train_upsample<-list()
  for (i in 1:5){
    t<-data.frame(rbind(r1[which(train1_folds==i),],r0[which(train0_folds==i),]))
    tup<-data.frame(rbind(r1[which(train1_folds==i),],r0[sample(which(train0_folds==i),nrow(r1[which(train1_folds==i),])),]))
    train[[i]]<-t
    train_upsample[[i]]<-tup
  }
remove(train0_folds,train1_folds)
   
   
#######################################################################################
# D2.prepare rf training models. use different mtry values to find correct ratio/best model


model_rf<-list()
perf.rf<-matrix(0,nrow = 5,ncol=10)
for(k in 1:5){
#  z<-floor((k+1)/2)
  trf<-  randomForest(x=train[[k]][,3:13], y=factor(train[[k]][,2]),mtry=1+k, sampsize = c('0'=1800,'1'=1800))
  model_rf[[k]]   <- trf
}

 confusion<-matrix(0,nrow = 2,ncol = 5)
 for (i in 1:5){
   for (j in 1:5){
         preds.rf.train <- predict(model_rf[[i]], newdata=train[[j]])
         perf.rf[i,j]<-(sum(preds.rf.train==train[[j]]$SeriousDlqin2yrs))/nrow(train[[j]])
         pred<-prediction(as.numeric(preds.rf.train),train[[j]][,2])
         perf_ROC=performance(pred,"tpr","fpr")
         confusion[1,j]<-perf_ROC@x.values[[1]][2]
         confusion[2,j]<-perf_ROC@y.values[[1]][2]
         }
   perf.rf[i,7]<-mean(confusion[1,1:5])
   perf.rf[i,8]<-mean(confusion[2,1:5])
   perf.rf[i,9]<- perf.rf[i,8]- perf.rf[i,7]
   perf.rf[i,6]<-mean(perf.rf[i,1:5])
 }

perf<-data.frame(perf.rf)
perf[,10]<-seq(2,6,1)
ggplot(perf, aes(x=perf$X10)) + xlab("mtry") + geom_line(aes(y=perf$X7,colour="fpr"))+ geom_line(aes(y=perf$X8,colour="tpr"))+ geom_line(aes(y=perf$X6,colour="accuracy"))+ geom_line(aes(y=perf$X9,colour="adj. tpr"))
perf_txt<-format(perf,digits=2)
htm_perf_rf<-htmlTable::htmlTable(perf_txt,header= c("set1","set2","set3","set4","set5","mean","fpr","tpr","adj. tpr","mtry"),rnames=c("model1","model2","model3","model4","model5"))

##################################################################################
#  validating best model performance

perf.rf.final<-matrix(0,nrow = 5,ncol=8)
confusion<-matrix(0,nrow = 2,ncol = 5)
for (i in 1:5){
  for (j in 1:5){
    preds.rf.train <- predict(model_rf[[5]], newdata=train[[j]])
    perf.rf.final[i,j]<-(sum(preds.rf.train==train[[j]]$SeriousDlqin2yrs))/nrow(train[[j]])
    pred<-prediction(as.numeric(preds.rf.train),train[[j]][,2])
    perf_ROC=performance(pred,"tpr","fpr")
    confusion[1,j]<-perf_ROC@x.values[[1]][2]
    confusion[2,j]<-perf_ROC@y.values[[1]][2]
  }
  perf.rf.final[i,7]<-mean(confusion[1,1:5])
  perf.rf.final[i,8]<-mean(confusion[2,1:5])
  perf.rf.final[i,6]<-mean(perf.rf.final[i,1:5])
  perf.rf.final[i,9]<- perf.rf[i,8]- perf.rf[i,7]
}

View(perf.rf.final)
perf<-data.frame(perf.rf.final)
perf_txt<-format(perf,digits=2)
htm_perf_rf<-htmlTable::htmlTable(perf_txt,header= c("set1","set2","set3","set4","set5","mean","fpr","tpr"),rnames=c("model1","model2","model3","model4","model5","model6","model7","model8","model9","model10","model11","model12","model13","model14","model15"))

#########################################################################################
# trying to balance class distribution using SMOTE
# took one of my training sets and evened out the classes using smote.
# now it's ~50K observations, a little big. so I chop it into CV-3,

smr0<-c()
smr1<-c()

for (i in 1:5){
  data1<-train[[i]][,2:13]
 data1$SeriousDlqin2yrs<-factor(data1$SeriousDlqin2yrs)
 bal_ds<-SMOTE(SeriousDlqin2yrs ~ .,data1,perc.over= 1000,perc.under=110)
 smr1<-data.frame(rbind(smr1,bal_ds[bal_ds$SeriousDlqin2yrs==1,]))
 smr0<-data.frame(rbind(smr0,bal_ds[bal_ds$SeriousDlqin2yrs==0,]))
}

 #k<-floor((nrow(smr0)+nrow(smr1))/30000)
 k<-5
 smtrain1_folds <- cut(seq(1,nrow(smr1)),breaks=5,labels=FALSE)
 smtrain0_folds <- cut(seq(1,nrow(smr0)),breaks=5,labels=FALSE)

 smtrain<-list()
 for (i in 1:k){
      t<-data.frame(rbind(smr1[which(smtrain1_folds==i),],smr0[which(smtrain0_folds==i),]))
      smtrain[[i]]<-t
     }
#remove(smtrain0_folds,smtrain1_folds)


##########################################################################################
#  train 7 models on the 7 SMOTE-balanced data sets , then run them over the "original" data sets 

  model_rf_sm<-list()
  perf.rf.sm<-matrix(0,nrow = 5,ncol=10)
   for(k in 1:5){
     trf<-  randomForest(x=smtrain[[k]][,2:12], y=smtrain[[k]][,1])
     model_rf_sm[[k]]   <- trf
   }
  perf.rf.sm<-matrix(0,nrow = 5,ncol=9)
   confusion<-matrix(0,nrow = 2,ncol = 7)
   for (i in 1:5){
     for (j in 1:5){
       preds.rf.train <- predict(model_rf_sm[[i]], newdata=train[[j]])
       perf.rf.sm[i,j]<-(sum(preds.rf.train==train[[j]]$SeriousDlqin2yrs))/nrow(train[[j]])
       pred<-prediction(as.numeric(preds.rf.train),train[[j]][,2])
       perf_ROC=performance(pred,"tpr","fpr")
       confusion[1,j]<-perf_ROC@x.values[[1]][2]
       confusion[2,j]<-perf_ROC@y.values[[1]][2]
     }
     perf.rf.sm[i,7]<-mean(confusion[1,1:5])
     perf.rf.sm[i,8]<-mean(confusion[2,1:5])
     perf.rf.sm[i,6]<-mean(perf.rf.sm[i,1:5])
     perf.rf.sm[i,9]<- perf.rf.sm[i,8]- perf.rf.sm[i,7]

   }

   perf<-data.frame(perf.rf.sm)
   perf_txt<-format(perf,digits=2)
   htm_perf_rf<-htmlTable::htmlTable(perf_txt,header= c("set1","set2","set3","set4","set5","mean","fpr","tpr","adj. TPR"),rnames=c("model1","model2","model3","model4","model5"))

# #########################################################################################
# # E. neural nets
# #E1. preparing normalised training and validation data sets for neural nets
# train_nn<-list()
# for (i in 1:5){
#   t<-data.frame(cbind(smtrain[[i]][,1],scale(smtrain[[i]][,2:12])))
#   train_nn[[i]]<-t
# }
#   
# 
# # E2. preparing neural net models with varying neuron number
# 
# set.seed(93837475)
# model_nn<-list()
# perf.nn<-matrix(0,nrow = 5,ncol=11)
# for(k in 1:5){
# #   z<-floor((k+1)/2)
#   tnn<-  nnet(x=train_nn[[k]][,2:12], y=class.ind(train_nn[[k]][,1]), size=4*k, softmax=TRUE,decay=5e-4, maxit=200)
#   model_nn[[k]]   <- tnn
# }
# 
# # E3. testing model accuracy using cross validation on the training sets

# perf.nn<-matrix(0,nrow = 5,ncol=9)
# confusion<-matrix(0,nrow = 2,ncol = 5)
# for (i in 1:5){
#   for (j in 1:5){
#       preds.nn.train<-factor(predict(model_nn[[i]], newdata=train[[j]][,3:12], type='class'))
#       perf.nn[i,j]<-(sum(preds.nn.train==train[[j]][,2]))/nrow(train[[j]])
#       pred<-prediction(as.numeric(preds.nn.train),train[[j]][,2])
#       perf_ROC=performance(pred,"tpr","fpr")
#       confusion[1,j]<-perf_ROC@x.values[[1]][2]
#       confusion[2,j]<-perf_ROC@y.values[[1]][2]
# 
#   }
#    perf.nn[i,7]<-mean(confusion[1,1:5])
#    perf.nn[i,8]<-mean(confusion[2,1:5])
#    perf.nn[i,6]<-mean(perf.nn[i,1:5])
#    perf.nn[i,9]<- perf.nn[i,8]- perf.nn[i,7]
# }
# 
#  perf<-data.frame(perf.nn)
#  perf_txt<-format(perf,digits=2)
#  htm_perf_rf<-htmlTable::htmlTable(perf_txt,header= c("set1","set2","set3","set4","set5","mean","fpr","tpr","adj. TPR"),rnames=c("model1","model2","model3","model4","model5","model6","model7"))

# ##########################################################################################
#   #  F. examining overfitting using Neural Networks
#   #  F1. generating varying scaled training sets for NN model
#   trn_size<-seq(10,1000, 10)
#   overfit.perf<-matrix(0,nrow=100,ncol=3)
#   for (sz in trn_size){
#     t<-data.frame(rbind(r1[sample(nrow(r1),p1*sz),],r2[sample(nrow(r2),p1*sz),]))
#     train_set<-data.frame(cbind(scale(t[,1:6],center = FALSE),t[,7]))
#     tnn<-  nnet(x=train_set[,1:6], y=class.ind(train_set[,7]), size=15, softmax=TRUE,decay=5e-4, maxit=100)
#     iter.perf<-matrix(0,nrow=10,ncol=2)
#     for (j in 1:10){
#       preds.nn.train<-factor(predict(tnn, newdata=train_nn[[j]][,1:6], type='class'))
#       preds.nn.test<-factor(predict(tnn, newdata=val_nn[[j]][,1:6], type='class'))
#       iter.perf[j,1]<-(sum(preds.nn.train==train_nn[[j]][,7]))/nrow(train_nn[[j]])
#       iter.perf[j,2]<-(sum(preds.nn.test==val_nn[[j]][,7]))/nrow(val_nn[[j]])
#     }
#     overfit.perf[which(trn_size==sz),1]<-sz
#     overfit.perf[which(trn_size==sz),2]<-mean(iter.perf[,1])
#     overfit.perf[which(trn_size==sz),3]<-mean(iter.perf[,2])
#   }
#   
#   
#   overfit.perf<-data.frame(overfit.perf)
#   colnames(overfit.perf)<-c("train_set_size","train_error", "test_error")
#   row.names(overfit.perf)[1:100]<-trn_size
#   ggplot(overfit.perf, aes(x=train_set_size))  + geom_line(aes(y=train_error,colour="train_error"))+ geom_line(aes(y=test_error,colour="test_error"))
