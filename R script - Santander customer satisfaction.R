
#Table of content

#1 Prediction 1
#1.1 Fast GBM
#1.2 Load libraries
#1.3 Reading in data 
#1.4 Data pre-processing
#1.5 Split train/test
#1.6 Fast GBM 1
#1.7 Fast GBM 2 
#1.8 Fast GBM 3
#1.9 Prediction

#2 Prediction 2 
#2.1 Auto modelling
#2.2 Imputation
#2.3 Hyper-parameter tuning
#2.4 Optimized predictive model GBM bernoulli
#2.5 Prediction

#3 Prediction 3
#3.1 GBM with all variables 
#3.2 Fast GBM 4 
#3.3 Prediction

#4 Prediction 4
#4.1 Ensembling
#4.2 Stacking ensemble

#5 Prediction 5
#5.1 EDA in-depth analysis
#5.2 Descriptive statistics
#5.3 The Y/response variable
#5.4 Correlation
#5.5 PCA full dataframe
#5.6 PCA reduced dataframe
#5.7 PCA PC's automatically selected

#6 Prediction 6
#6.1 Lean XG Boost
#6.2 Prediction

######################################################################################################################
######################################################################################################################
##1) Prediction 1  

#1.1) Fast GBM 

#First we will apply a fast GBM for quick first results and valuable insight in the data.

######################################################################################################################
######################################################################################################################
##1.2) Load libraries

set.seed(4444)

install.packages("")

library(gbm)
library(cvAUC)
library(randomForest)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(MASS)
library(caret)
library(RANN)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(pROC)
library(ggplot2)
library(reshape)
library(mlbench)
library(MLmetrics)
library(Hmisc)
library(mlbench)
library(e1071)
library(sqldf)
library(corrplot)
library(xgboost)
library(Matrix)
library(magrittr)

######################################################################################################################
######################################################################################################################
##1.3) Reading in data 

data <- read.csv(file="C:/Santander customer Satisfaction/data.csv")

######################################################################################################################
######################################################################################################################
##1.4) Data pre-processing

# Randomizing the dataset
#data<- data[sample(1:nrow(data)),]

#Subset data
data <- data[1:20000,]

#Checking data type response variable
is.numeric(data$TARGET)

#Creating clean dataset
myvars <- c("TARGET","var3","var15","var38")
data1 <- data[myvars]

#Count the na's
NAcol <- which(colSums(is.na(data1)) > 0)
cat('There are', length(NAcol), 'columns with missing values')

#Histogram response/dependent variable
ggplot(data1, aes(TARGET)) +
  geom_bar(fill = "#0073C2FF")

######################################################################################################################
######################################################################################################################
##1.5) Split train/test

# Split train/test
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train <- data1[partition,]
test <- data1[-partition,]

######################################################################################################################
######################################################################################################################
##1.6) Fast GBM 1

# GBM 1 Bernoulli
gbm1 <- gbm(TARGET~.,                 
           data=train,                
           distribution='bernoulli',  
           n.trees=300,               
           shrinkage=0.1,             
           interaction.depth=1,       
           train.fraction = 1,        
           cv.fold = 5,
           n.minobsinnode = 10,       
           keep.data = TRUE,          
           verbose=F,                 
           n.cores = NULL)  #Using all cores                

# plot the performance
best.iter<- gbm.perf(gbm1,method="cv")   
print(best.iter)

#The marginal effect plots 
for(i in 1:length(gbm1$var.names)){
  plot(gbm1, i.var = i
       , ntrees= gbm.perf(gbm1, plot.it = FALSE)
       , type = "response"
  )
}

summary(gbm1,n.trees=best.iter) 

######################################################################################################################
######################################################################################################################
##1.7) Fast GBM 2 

#Exclude the ID column
myvars <- names(data) %in% c('ID') 
data <- data[!myvars]

# Split train/test
partition <- createDataPartition(data$TARGET, p=0.7, list=FALSE)
train <- data[partition,]
test <- data[-partition,]

# GBM 2 Bernoulli
gbm1 <- gbm(TARGET~.,               
            data=train,                
            distribution='bernoulli',  
            n.trees=300,               
            shrinkage=0.1,             
            interaction.depth=1,       
            train.fraction = 1,        
            cv.fold = 5,
            n.minobsinnode = 10,       
            keep.data = TRUE,          
            verbose=F,                 
            n.cores = NULL)  #Using all cores              

# Cross-validation performance
best.iter <- gbm.perf(gbm1,method="cv")
print(best.iter)

#The marginal effect plots 
for(i in 1:length(gbm1$var.names)){
  plot(gbm1, i.var = i
       , ntrees= gbm.perf(gbm1, plot.it = FALSE)
       , type = "response"
  )
}

summary(gbm1)

######################################################################################################################
######################################################################################################################
##1.8) Fast GBM 3 

#Creating clean dataset
myvars <- names(data) %in% c("TARGET","var15","var38","saldo_var30")
data1 <- data[myvars]

# Split train/test
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train <- data1[partition,]
test <- data1[-partition,]

#1 GBM 3 Bernoulli
gbm1 <- gbm(TARGET~.,                 
            data=train,                
            distribution='bernoulli',  
            n.trees=200,               
            shrinkage=0.1,             
            interaction.depth=1,       
            train.fraction = 1,        
            cv.fold = 5,
            n.minobsinnode = 10,       
            keep.data = TRUE,          
            verbose=F,                 
            n.cores = NULL)  #Using all cores           

# plot the performance
best.iter<- gbm.perf(gbm1,method="cv")   
print(best.iter)

#The marginal effect plots 
for(i in 1:length(gbm1$var.names)){
  plot(gbm1, i.var = i
       , ntrees= gbm.perf(gbm1, plot.it = FALSE)
       , type = "response"
  )
}

summary(gbm1,n.trees=best.iter) 

######################################################################################################################
##1.9 Prediction 

# f.predict on canonical scale 
f.predict <- predict.gbm(gbm1, test, n.trees = best.iter, type='link')
head(f.predict)
str(f.predict)
#OR
f.predict <- predict.gbm(gbm1, test, n.trees = best.iter)
head(f.predict)
str(f.predict)

# transform to probability scale 
p.pred <- 1/(1+exp(-f.predict))
head(p.pred)
str(p.pred)
#OR
# prediction on probability scale
prediction <- predict(gbm1, test, n.trees = best.iter, type='response')
head(prediction)
str(prediction)

test$TARGET <- as.factor(test$TARGET)
is.factor(test$TARGET)

density(prediction) %>% plot

# Prediction on test set
preds <- predict(gbm1, newdata = test, n.trees = best.iter)
labels <- test[,"TARGET"]

# Computing AUC 
cvAUC::AUC(predictions = preds, labels = labels)

test$TARGET <- as.numeric(test$TARGET)

######################################################################################################################
######################################################################################################################
##2) Prediction 2 
                            
#2.1) Auto-modelling 
                            
#First we'll apply a fast GBM for quick first results and valuable insight in the data.                            
######################################################################################################################
######################################################################################################################
##2.2 Imputation

set.seed(4444)

#Checking for Na's in dataframe
NAcol <- which(colSums(is.na(data1)) > 0)
apply(data1, 2, function(x) any(is.na(x)))
cat('There are', length(NAcol), 'columns with missing values')

######################################################################################################################
######################################################################################################################
#2.3) Hyper parameter tuning

#Creating clean dataset
myvars <- names(data) %in% c("TARGET","var15","saldo_var30")
data1 <- data[myvars]

# Split train/test
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train <- data1[partition,]
test <- data1[-partition,]

#Converting the outcome/response to factor
train$TARGET <- as.factor(train$TARGET)
is.factor(train$TARGET)

## 3.4 Parameter Tuning
fitControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1)

model_gbm<-train(TARGET~.,method='gbm',data=train,distribution="bernoulli",verbose=F,trControl=fitControl,tuneLength=10)

print(model_gbm)

plot(model_gbm)

## 3.4 Parameter Tuning
fitControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1)

#Creating grid
grid <- expand.grid(n.trees=c(50,100,150),
                    shrinkage=c(0.01,0.05,0.1),
                    n.minobsinnode = c(10),
                    interaction.depth=c(1,2,3))

# training the model
model_gbm<-train(TARGET~.,method='gbm',trControl=fitControl,distribution="bernoulli",verbose=F,data=train,tuneGrid=grid)

print(model_gbm)

plot(model_gbm)

######################################################################################################################
######################################################################################################################
#2.4) Optimized predictive model GBM Bernoulli

## 3.4 Parameter Tuning
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1)

# Creating grid
grid <- expand.grid(n.trees=c(100),
                    shrinkage=c(0.01),
                    n.minobsinnode = c(10),
                    interaction.depth=c(1))

# training the model
model_gbm<-train(TARGET~.,method='gbm',distribution="bernoulli",trControl=fitControl,verbose=F,data=train,tuneGrid=grid)

print(model_gbm)

summary(model_gbm)
#OR
plot(varImp(object=model_gbm),main="GBM - Variable Importance")

getTrainPerf(model_gbm)

confusionMatrix(model_gbm)

######################################################################################################################
##2.5) Prediction  

gbmpred <- predict(model_gbm, test)
head(gbmpred)
str(gbmpred)
#OR
gbmpred <- predict(model_gbm, test, type = "raw")
head(gbmpred)
str(gbmpred)

gbmprobs <- predict(model_gbm, test, type = "prob")
head(gbmprobs)
str(gbmprobs)

test$TARGET <- as.factor(test$TARGET)
is.factor(test$TARGET)

gbmprobs$obs = test$TARGET
mnLogLoss(gbmprobs, lev = levels(gbmprobs$obs))

postResample(gbmpred, test$TARGET)

caret::confusionMatrix(gbmpred, test$TARGET)

rocCurve <- roc(response = test$TARGET,
                predictor = gbmprobs[, "1"],
                levels = rev(levels(test$TARGET)))
rocCurve
plot(rocCurve)
#plot(rocCurve,
     #print.thres = c(.5,.2),
     #print.thres.pch = 16,
     #print.thres.cex = 1.2)

mPred = predict(model_gbm, test, na.action = na.pass)

mResults = predict(model_gbm, test, na.action = na.pass, type = "prob")
mResults$obs = test$TARGET

mnLogLoss(mResults, lev = levels(mResults$obs))

gbmprobs$pred = predict(model_gbm, test, na.action = na.pass)
multiClassSummary(gbmprobs, lev = levels(gbmprobs$obs))

evalResults <- data.frame(Class = test$TARGET)
evalResults$GBM <- predict(model_gbm, test, na.action = na.pass, type = "prob")

head(evalResults)

######################################################################################################################
######################################################################################################################
##3) Prediction 3 

#3.1) GBM with all effecting variables. 

######################################################################################################################
######################################################################################################################
##3.2 Fast GBM 4

set.seed(4444)

#Creating clean dataset
myvars <- c("TARGET","var38","num_var22_ult1","saldo_var30","num_var22_hace2","num_var22_ult3",
            "saldo_var37","saldo_medio_var5_hace2","num_var45_hace2","num_var45_ult1","saldo_var5",
            "num_var45_ult3","num_var45_hace3")
data1 <- data[myvars]

# Split train/test
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train <- data1[partition,]
test <- data1[-partition,]

train$TARGET <- as.numeric(train$TARGET)
is.numeric(train$TARGET)

#1 GBM 1 Bernoulli
gbm1 <- gbm(TARGET~.,                  
            data=train,                
            distribution='bernoulli',   
            n.trees=300,                
            shrinkage=0.1,              
            interaction.depth=1,        
            train.fraction = 1,         
            cv.fold = 5,
            n.minobsinnode = 10,        
            keep.data = TRUE,           
            verbose=F,                  
            n.cores = NULL)                             

# 5-fold cross-validation
best.iter <- gbm.perf(gbm1,method="cv")
print(best.iter)

#The marginal effect plots 
for(i in 1:length(gbm1$var.names)){
  plot(gbm1, i.var = i
       , ntrees= gbm.perf(gbm1, plot.it = FALSE)
       , type = "response"
  )
}

summary(gbm1)

######################################################################################################################
##3.3 Prediction

predict(gbm1, newdata = head(test), type = "response")

library(magrittr)
density(prediction) %>% plot

# Prediction on test set
preds <- predict(gbm1, newdata = test, n.trees = best.iter)
labels <- test[,"TARGET"]

# Computing AUC 
cvAUC::AUC(predictions = preds, labels = labels)

######################################################################################################################
######################################################################################################################
##4) Prediction 4 

#4.1) Ensembling 

#For the fourth prediction we'll use an ensemble model.

######################################################################################################################
######################################################################################################################
##4.2) Stacking ensemble

set.seed(4444)

#Creating clean dataset
myvars <- names(data) %in% c("TARGET","var15","var38","saldo_var30")
data1 <- data[myvars]

# Split train/test
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train2 <- data1[partition,]
test2 <- data1[-partition,]

train2$TARGET[train$TARGET=="0"] <- "No"
train2$TARGET[train$TARGET=="1"] <- "Yes"

# create the base models
control <- trainControl(method="repeatedcv", number=2, repeats=1, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('glm', 'xgbTree','rf')
set.seed(100)
models <- caretList(TARGET~., data=train2, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

modelCor(results)
splom(results)

set.seed(100)

# stacking using gbm
stackControl <- trainControl(method="repeatedcv", number=5, repeats=1, savePredictions=TRUE, classProbs=TRUE)
set.seed(100)
stack.glm <- caretStack(models, method="gbm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

gbmpred <- predict(stack.glm, test2)
head(gbmpred)
str(gbmpred)
#OR
gbmpred <- predict(stack.glm, test2, type = "raw")
head(gbmpred)
str(gbmpred)

gbmprobs <- predict(stack.glm, test2, type = "prob")
head(gbmprobs)
str(gbmprobs)

######################################################################################################################
######################################################################################################################
##5) Prediction 5

#5.1) EDA In-depth analysis

#For the fifth prediction we'll first do an in-depth EDA and then make predictions,

######################################################################################################################
######################################################################################################################
##5.2 Descriptive statistics

set.seed(4444)

#Exclude the ID column
myvars <- names(data) %in% c('ID') 
data <- data[!myvars]

#Descriptive statistics
head(data, n=10)

dim(data)

str(data)

summary(data)

Hmisc::describe(data)

sapply(data, class)

sapply(data, mean, na.rm=TRUE)

sapply(data[,1:10], sd)

skew <- apply(data[,1:10], 2, skewness)
print(skew)

#Check NA's
apply(data, 2, function(x) any(is.na(x)))
NAcol <- which(colSums(is.na(data)) > 0)
cat('There are', length(NAcol), 'columns with missing values')

#Check class Y (balanced/unbalanced)
y <- data$TARGET
cbind(freq=table(y), percentage=prop.table(table(y))*100)

######################################################################################################################
######################################################################################################################
##5.3 The Y/response variable

ggplot(data, aes(TARGET)) +
  geom_bar(fill = "#0073C2FF")


sqldf("SELECT TARGET, COUNT(TARGET) AS count FROM data 
      GROUP BY TARGET")

prop.table(table(data$TARGET))

data$TARGET <- as.numeric(data$TARGET)

d <- density(data$TARGET)
plot(d, main="TARGET")
polygon(d, col="lightblue", border="black")

x <- data$TARGET 
h<-hist(x, breaks=10, col="lightblue", xlab="Miles Per Gallon", 
        main="Histogram with Normal Curve") 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="darkblue", lwd=2)

######################################################################################################################
######################################################################################################################
##5.4 Correlation

# Correlations with Y=Target
correlations <- cor(data[,1:369],data[,370])
print(correlations)

# Correlation matrix
correlationMatrix <- cor(data[,1:10])
print(correlationMatrix)
corrplot(correlationMatrix, method="circle",col="darkblue",tl.col = "black")

# Deleting all constant features 
cat("\n## Removing the constants features.\n")
for (f in names(data)) {
  if (length(unique(data[[f]])) == 1) {
    cat(f, "is constant in data. Deleting this feature.\n")
    data[[f]] <- NULL
  }
}

# Deleting all identical features 
features_pair <- combn(names(data), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(data[[f1]] == data[[f2]])) {
      cat(f1, "and", f2, "are identical.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(data), toRemove)

data <- data[, feature.names]

# Find high correlating features
correlationMatrix <- cor(data[,1:278])
highCorrelation <- findCorrelation(correlationMatrix, cutoff=0.75)
# printing indexes high correlating features
print(highCorrelation)
# printing names high correlating features
names(data)[highCorrelation]

# Correlationmatrix Y <- X features
correlationMatrix <- cor(data[,1:20],data[,278])
print(correlationMatrix)
corrplot(correlationMatrix, method="circle",col="darkblue",tl.col = "black")

######################################################################################################################
######################################################################################################################
##5.5 PCA 1 - Full dataframe 

#Split the data 
partition <- createDataPartition(data$TARGET, p=0.7, list=FALSE)
train <- data[partition,]
test <- data[-partition,]

#remove the dependent/outcome variable
pca.train <- subset(train, select = -c(TARGET))
pca.test <- subset(test, select = -c(TARGET))

for (f in names(pca.train)) {
  if (length(unique(pca.train[[f]])) == 1) {
    pca.train[[f]] <- NULL
  }
}

#Variables in pca.train
colnames(pca.train)

#Variable class
str(pca.train)

prin_comp <- prcomp(pca.train, scale. = T)
names(prin_comp)

prin_comp$center

prin_comp$scale

prin_comp$rotation

prin_comp$rotation[1:5,1:4]

dim(prin_comp$x)

std_dev <- prin_comp$sdev

pr_var <- std_dev^2

pr_var[1:10]

prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "The proportion of explained variance",
       type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "The cumulative Proportion of explained variance",
       type = "b")

#adding a training set with principal components
pca.train <- data.frame(TARGET = train$TARGET, prin_comp$x)

#Distilling first 50 PCAs
pca.train <- pca.train[,1:51]

#Rpart
library(rpart)
rpart.model <- rpart(TARGET ~ .,data = pca.train, method = "anova")
rpart.model

#transforming test into PCA
pca.test <- predict(prin_comp, newdata = pca.test)
pca.test <- as.data.frame(pca.test)

#Distilling first 50 components
pca.test <- pca.test[,1:51]

#Pprediction on pca.test 
pca.pred1 <- predict(rpart.model, pca.test)
head(pca.pred1)

######################################################################################################################
######################################################################################################################
##5.6 PCA 2 - Reduced dataframe

# Finding most important set features
control <- trainControl(method="repeatedcv", number=2, repeats=1)
model <- train(TARGET~., data=data, method="gbm", preProcess="scale", verbose=F, trControl=control)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

# Creating reduced dataset
myvars <- names(data) %in% c("var15","saldo_var30","var38","imp_op_var41_efect_ult1",
                             "num_var45_ult1","num_var45_ult1","ind_var8_0",
                             "num_meses_var5_ult3","saldo_medio_var5_hace2",
                             "num_op_var41_efect_ult1","ind_var30_0","saldo_medio_var8_hace2",
                             "imp_op_var41_efect_ult3","var3","saldo_medio_var5_ult1",
                             "saldo_var37","num_op_var39_efect_ult3","var36",
                             "imp_op_var39_efect_ult1","saldo_medio_var5_hace3",
                             "num_var45_hace2","TARGET")
data1 <- data[myvars]

# Correlation matrix
correlationMatrix <- cor(data1[,1:21])
print(correlationMatrix)
corrplot(correlationMatrix, method="circle",col="darkblue",tl.col = "black")

# Splitting the data 
partition <- createDataPartition(data1$TARGET, p=0.7, list=FALSE)
train <- data1[partition,]
test <- data1[-partition,]

#removing the dependent variable
pca.train <- subset(train, select = -c(TARGET))
pca.test <- subset(test, select = -c(TARGET))

prin_comp <- prcomp(pca.train, scale. = T)

#Plotting principal components.
biplot(prin_comp, scale = 0, col="darkblue")

#PCA prep
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
pr_var[1:10]
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#Adding a training set with principal components
pca.train <- data.frame(TARGET = train$TARGET, prin_comp$x)

#Distilling the first 15 PCA's
pca.train <- pca.train[,1:16]

#Rpart
rpart.model <- rpart(TARGET ~ .,data = pca.train, method = "anova")
rpart.model

#transforming test into PCA
pca.test <- predict(prin_comp, newdata = pca.test)
pca.test <- as.data.frame(pca.test)

#Distilling the first 15 components
pca.test <- pca.test[,1:16]

#Prediction on pca.test
pca.pred2 <- predict(rpart.model, pca.test)
head(pca.pred2)

######################################################################################################################
######################################################################################################################
##5.6 PCA 3 - PCA PC's automatically selected

control <- rfeControl(functions=rfFuncs, method="cv", number=2)
# RFE algorithm
results <- rfe(pca.train[,2:15], pca.train[,1], sizes=c(1:20), rfeControl=control)

print(results)
# list features
predictors(results)
# plotting the result
plot(results, type=c("g", "o"))

pca.train$TARGET <- as.factor(pca.train$TARGET)

control <- trainControl(method="repeatedcv", number=2, repeats=1)
# training the model
model <- train(TARGET ~ PC8, data=pca.train, method="gbm",verbose=F, trControl=control)

print(model)

######################################################################################################################
##5.7 Prediction

# PCA prediction 1
pca.pred1 <- predict(rpart.model, pca.test)
head(pca.pred1)

# PCA prediction 2
pca.pred2 <- predict(rpart.model, pca.test)
head(pca.pred2)

# PCA prediction 3
pca.pred3 <- predict(model)
head(gbmpred)

pca.probs <- predict(model, test, type = "prob")
head(gbmprobs)

######################################################################################################################
######################################################################################################################
##6) Prediction 6 - Lean XGB prediction

#6.1) Lean XGB prediction

# For the sixth prediction we'll do a lean prediction with the often best performing
# algorithm XG Boost.

######################################################################################################################
######################################################################################################################

set.seed(4444)

data <- read.csv(file="C:/Santander customer Satisfaction/data.csv")

myvars <- names(data) %in% c('ID') 
data <- data[!myvars]

for (f in names(data)) {
  if (length(unique(data[[f]])) == 1) {
    data[[f]] <- NULL
  }
}

features_pair <- combn(names(data), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(data[[f1]] == data[[f2]])) {
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(data), toRemove)

data <- data[, feature.names]

partition <- createDataPartition(data$TARGET, p=0.7, list=FALSE)
train <- data[partition,]
test <- data[-partition,]

train.y <- train$TARGET
train$TARGET <- NULL
train$TARGET <- train.y

train <- sparse.model.matrix(TARGET ~ ., data = train)

ltrain <- xgb.DMatrix(data=train, label=train.y)
watchlist <- list(train=ltrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 3,
                subsample           = 0.5,
                colsample_bytree    = 0.6
)

clf <- xgb.train(   params              = param, 
                    data                = ltrain, 
                    nrounds             = 500, 
                    verbose             = 2,
                    watchlist           = watchlist,
                    maximize            = F
)

test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

######################################################################################################################
##5.7 Prediction

preds <- predict(clf, test)
head(preds)

# Etc.#

######################################################################################################################
######################################################################################################################

# Optimization 1: 

# Optimization 2: 

# Optimization 3: 

# etc.
