rm(list = ls())
#set the path
setwd("D:\\frauddetection")
getwd()


#Read the Train and Test Data
loan=read.csv("Loan_data_new.csv",na.strings = "")
str(loan)
names(loan)
#To get only row percentages
library(gmodels)
CrossTable(loan$Gender,loan$Loan_Status,prop.r=T,prop.c=F,prop.t=F,prop.chi=F)

#To get only row percentages with chisquare Test
#H0: There is no association b/w Gender and Loan_Status (Two categorical Vars)
#H1: There is an association b/w Gender and Loan_Status (Two categorical Vars)


CrossTable(loan$Married,loan$Loan_Status,prop.r=T,prop.c=F,prop.t=F,prop.chi=F)

#-------------------------------------------------------------------------------------------
CrossTable(loan$Gender,loan$Loan_Status,prop.r=T,prop.c=F,prop.t=F,prop.chi=F,chisq=T)
CrossTable(loan$Married,loan$Loan_Status,prop.r=T,prop.c=F,prop.t=F,prop.chi=F,chisq=T)

summary(loan)

#To find out the missing values by variable wise
sapply(loan, function(x) sum(is.na(x)))

#Dependents - 90% missing info (remove it from the analysis)

#Check the missing values % by variable wise
library(VIM)
aggr_plot <- aggr(loan, col=c('navyblue','red'),
                  numbers=TRUE, sortVars=TRUE, 
                  labels=names(loan), cex.axis=.7, 
                  gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))

#Impute the missing values
library(mice)
#imputed_Data <- mice(loan[,-c(1,4)], m = 3, method = vector("character", length = ncol(loan[,-c(1,4)])),
#                     predictorMatrix = (1 - diag(1, ncol(loan[,-c(1,4)]))),
#                     visitSequence = (1:ncol(loan[,-c(1,4)]))[apply(is.na(loan[,-c(1,4)]), 2, any)],
#                     form = vector("character", length = ncol(loan[,-c(1,4)])),
#                     post = vector("character", length = ncol(loan[,-c(1,4)])), defaultMethod = c("pmm",
#                     "logreg", "polyreg", "polr"), maxit = 5, diagnostics = TRUE,MaxNWts = 2000)


class(loan$Credit_History)

loan$Credit_History<-as.factor(loan$Credit_History)

class(loan$Credit_History)

names(loan)
imputed_Data <- mice(loan[,-c(1,4)], m = 3, method = "cart")
#Save the data in a dataframe
dt_imp <- complete(imputed_Data)
summary(dt_imp)

dim(dt_imp)
#
####Final data prep has been completed
#--------------------------------------------------------------------
#Training and Testing Process
#--------------------------------------------------------------------
#Divide the data into train and test
set.seed(123)
library(caTools)
split <- sample.split(dt_imp$Loan_Status ,SplitRatio=0.8)
train <- subset(dt_imp, split==T)
test <- subset(dt_imp, split==F)


CrossTable(dt_imp$Loan_Status)
CrossTable(train$Loan_Status)
CrossTable(test$Loan_Status)

names(train)
dim(train)


#Convert categorical variables into  factors
for (i in c(1:4,9:11)){
  train[,i]=as.character(train[,i])
}

for (i in c(1:4,9:11)){
  test[,i]=as.character(test[,i])
}

#Convert in to numeric
for (i in c(5:8)){
  train[,i]=as.numeric(train[,i])
}

for (i in c(5:8)){
  test[,i]=as.numeric(test[,i])
}



#logistic regression model on Loan_Status Vs all independent vars
m<-glm(train$Loan_Status=="Y"~.,data = train,family = binomial(link=logit))
summary(m)

#Check the accuracy of the model by using confusion matrix
library(caret)

Pred_train<-predict(m,train,type = "response")
Pred_train
summary(Pred_train)

train$Pred_Status = ifelse(Pred_train>0.8,1,0)

CrossTable(train$Loan_Status,train$Pred_Status,prop.r = T,prop.c = F,prop.t = F,prop.chisq = F)

#The accuracy of the model on train data 80% at 0.75 threshold


#Predict the probs on test
Pred_test<-predict(m,test,type = "response")
Pred_test

test$Pred_Status = ifelse(Pred_test>0.8,1,0)


CrossTable(test$Loan_Status,test$Pred_Status,prop.r = T,prop.c = F,prop.t = F,prop.chisq = F)
#The accuracy of the model on test data 77% at 0.75 threshold

