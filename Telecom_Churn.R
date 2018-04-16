##################Telecom Churn Analysis and Solution###########
################################################################
#Business Understanding
#Data Understanding
#Data Preparation & EDA
#Model Building 
#Model Evaluation
#Model deployment
################################################################

### Business Understanding:

# Based on the past and current customer information,
# the company has maintained a database containing personal/demographic information,
# the services availed by a customer and the expense information related to each customer.

## Objective:

# The aim is to automate the process of predicting 
# if a customer would be churn or not and to find the factors affecting the customer churn. 

# Whether a customer will churn or not will depend on data from the following three buckets:

# 1. Demographic Information
# 2. Services Availed by the customer
# 3. Overall Expenses

################################################################

### Data Understanding

# Install and Load the required packages
#install.packages("MASS")
#install.packages("car")
#install.packages("e1071")
#install.packages("caret", dependencies = c("Depends", "Suggests"))
#install.packages("cowplot")
#install.packages("GGally")
#install.packages("DescTools")
library(DescTools)
library(MASS)
library(car)    #Multicllinearity
library(e1071)
library(caret)
library(ggplot2)
library(cowplot)
library(caTools)
library(gridExtra)
library(grid)
library(caret)
################################################################
#############FINAL data

setwd("D:\\ML\\Regression_LoanData")
telecom_data = read.csv("TELECOM_RANDOM_data.csv",stringsAsFactors = F)
names(telecom_data)
str(telecom_data)
library(gmodels)
dim(telecom_data)
names(telecom_data)
sum(is.na(telecom_data))

# converting target variable telecom from No/Yes character to factor with levels 0/1 

telecom_data$Churn<- as.character(telecom_data$Churn)

#for continuous variables

telecom_con<- telecom_data[,c(2,7,8)]
names(telecom_con)

#for categorical variables
fact_variable <- data.frame(sapply(telecom_data, function(x) factor(x)))
str(fact_variable)
dim(fact_variable)

fact_variable$tenure = NULL
fact_variable$MonthlyCharges = NULL
fact_variable$TotalCharges = NULL

fact_variable$customerID = as.character(fact_variable$customerID)
fact_variable$Churn = as.character(fact_variable$Churn)

Churn = fact_variable$Churn
customerID = fact_variable$customerID

fact_variable$Churn = NULL
fact_variable$customerID = NULL

# Normalising continuous features 
telecom_con$tenure<- scale(telecom_con$tenure) 
telecom_con$MonthlyCharges<- scale(telecom_con$MonthlyCharges) 
telecom_con$TotalCharges<- scale(telecom_con$TotalCharges) 


dim(fact_variable)
# creating dummy variables for factor attributes
dummies<- data.frame(sapply(fact_variable, 
                            function(x) data.frame(model.matrix(~x-1,data =fact_variable))[,-1]))
names(dummies)

# Final dataset

telecom_final<- cbind(Churn,dummies,telecom_con)
str(telecom_final)
names(telecom_final)


telecom_final$customerID= telecom_data$customerID

telecom_final$Churn = factor(ifelse(telecom_final$Churn=="Yes",1,0))
dim(telecom_final)
#View(telecom_final)

# splitting the data between train and test
set.seed(1000)
sample_data = sample(2,nrow(telecom_final),replace = T,prob = c(0.7,0.3))

telecom_training = telecom_final[sample_data==1,]
dim(telecom_training)
telecom_test = telecom_final[sample_data==2,]
dim(telecom_test)

names(telecom_training)
telecom_training=telecom_training[,-34]
#telecom_test=telecom_test[,-1]
#################
names(telecom_training)

m1 = glm(Churn ~ ., data = telecom_training,family = binomial(link=logit))
summary(m1)

#To get only the significant variables
library("MASS")
model_2<- stepAIC(m1, direction="both")
summary(model_2)


m2 =glm(formula = Churn ~ Contract.xMonth.to.Month + Contract.xOne.year + 
          Contract.xTwo.year + PaymentMethod.xElectronic.check + PaymentMethod.xMailed.check + 
          gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xNo.internet.service + 
          OnlineSecurity.xYes + DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingTV.xYes + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m2)

#### Below variables are not sig.
# PaymentMethod.xMailed.check 
# OnlineSecurity.xYes

m3 =glm(formula = Churn ~ Contract.xMonth.to.Month + Contract.xOne.year + 
          Contract.xTwo.year + PaymentMethod.xElectronic.check +  
          gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xNo.internet.service + 
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingTV.xYes + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m3) # Still below variable having not sig.

# PaymentMethod.xElectronic.check

m4 =glm(formula = Churn ~ Contract.xMonth.to.Month + Contract.xOne.year + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xNo.internet.service + 
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingTV.xYes + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m4) ### Looks all the variables having Sig.
## Let me check the VIF
vif(m4) # Below variables are having High VIF

# Contract.xMonth.to.Month
# Contract.xOne.year
# Contract.xTwo.year
# StreamingTV.xNo.internet.service
# StreamingTV.xYes

### Remove the below variables due to High VIF and low Sig.
  # Contract.xOne.year

m5 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xNo.internet.service + 
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingTV.xYes + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m5) #### Looks Like Below Variables having sig
vif(m5) ### StreamingTV.xYes is having highest VIF

m6 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + OnlineSecurity.xNo.internet.service + 
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m6) # Below OnlineSecurity.xNo.internet.service 

m7 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + MultipleLines.xYes.phone.service + 
          InternetService.xFiber.optic + InternetService.xNo + 
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m7) # Looks all the variables having high sig
vif(m7)
# Below variables are having very low sig in the data

# MultipleLines.xYes.phone.service
# InternetService.xNo

m8 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + 
          InternetService.xFiber.optic +  
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingMovies.xYes + 
          IntrenetIssue + TotalCharges, family = binomial(link = logit), 
        data = telecom_training)

summary(m8) ## looks like all variables having above 99% of Sig.
VIF(m8) 
##### TotalCharges having 99% of Sig.

m9 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + 
          InternetService.xFiber.optic +  
          DeviceProtection.xYes + TechSupport.xYes + 
          StreamingTV.xNo.internet.service + StreamingMovies.xYes + 
          IntrenetIssue , family = binomial(link = logit),data = telecom_training)

summary(m9) ## all Variables are having 100% sig.
### Let me check the VIF
vif(m9)

# TechSupport.xYes
m10 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
          Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + 
          InternetService.xFiber.optic +  
          DeviceProtection.xYes +  
          StreamingTV.xNo.internet.service + StreamingMovies.xYes + 
          IntrenetIssue , family = binomial(link = logit),data = telecom_training)

summary(m10)
vif(m10) 

# StreamingTV.xNo.internet.service having Highest VIF

m11 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
           Contract.xTwo.year + gender + SeniorCitizen + Partner + MultipleLines.xYes + 
           InternetService.xFiber.optic +  
           DeviceProtection.xYes +  
           StreamingMovies.xYes + 
           IntrenetIssue , family = binomial(link = logit),data = telecom_training)

summary(m11)
vif(m11) #### Below variables 
# Contract.xTwo.year having the High VIF

m12 =glm(formula = Churn ~ Contract.xMonth.to.Month + 
           gender + SeniorCitizen + Partner + MultipleLines.xYes + 
           InternetService.xFiber.optic +  
           DeviceProtection.xYes +  
           StreamingMovies.xYes + 
           IntrenetIssue , family = binomial(link = logit),data = telecom_training)

summary(m12)
vif(m12)

## No Problem in VIF and Sig. Consider as a final model

final_model=m12

# (Intercept)                   < 2e-16 ***
#  Contract.xMonth.to.Month     0.000449 ***
#  gender                        < 2e-16 ***
#  SeniorCitizen                 < 2e-16 ***
#  Partner                      0.000157 ***
#  MultipleLines.xYes            < 2e-16 ***
#  InternetService.xFiber.optic  < 2e-16 ***
#  DeviceProtection.xYes         < 2e-16 ***
#  StreamingMovies.xYes          < 2e-16 ***
#  IntrenetIssue                 < 2e-16 ***


#########################################################################################
# Let's Choose the cutoff value. 
# Let's find out the optimal probalility cutoff

pred = predict(m12,telecom_test,type = "response")
head(pred)
head(telecom_test$Churn)
summary(telecom_test$Churn)

test_pred_churn <- factor(ifelse(pred >= 0.50, "Yes", "No"))
test_actual_churn <- factor(ifelse(telecom_test$Churn==1,"Yes","No"))

test_conf <- confusionMatrix(test_pred_churn, test_actual_churn, positive = "Yes")
test_conf

#   Accuracy : 0.9209
#   Sensitivity : 0.8380          
#   Specificity : 0.9489 
#########################################################################################
# Let's Choose the cutoff value. 
#

# Let's find out the optimal probalility cutoff

pred = predict(m12,telecom_test,type = "response")

perform_fn <- function(cutoff) 
{
  predicted_churn <- factor(ifelse(pred >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_churn, test_actual_churn, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}

# Creating cutoff values from 0.003575 to 0.812100 for plotting and initiallizing a matrix of 100 X 3.

# Summary of test probability

summary(pred)

s = seq(.01,.80,length=100)

OUT = matrix(0,100,3)


for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
}


plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()

legend(0.25,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))


cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)]
#cutoff <- max(abs(OUT[,1]-OUT[,2]))

cutoff #### 0.3132323
###########################
################ Find out the optimal Cutoff Value ##########

# Let's find out the optimal probalility cutoff 
# First let's create a function to find the accuracy, sensitivity and specificity
# for a given cutoff

require(ROCR)

pred <- prediction(pred, test_actual_churn)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf,
     colorize=T,
     main= "AUC Curve-Logit Model",
     ylab = "Sensitivity",
     xlab = "1-Specificity" )
abline(a=0, b= 1)

### AUC 


auc <- performance(pred, measure = "auc")
auc <- unlist(slot(auc, "y.values"))
auc = round(auc, 2) #AUC = 0.9380087  Good Model
auc <- paste(c("Auc ="),round(as.numeric(performance(pred,"auc")@y.values),digits=2),sep="")
legend("topleft",auc, bty="n")


##### partial ROC curve #### 

pROC = function(pred, fpr.stop){
  perf <- performance(pred,"tpr","fpr")
  for (iperf in seq_along(perf@x.values)){
    ind = which(perf@x.values[[iperf]] <= fpr.stop)
    perf@y.values[[iperf]] = perf@y.values[[iperf]][ind]
    perf@x.values[[iperf]] = perf@x.values[[iperf]][ind]
  }
  return(perf)
}

proc.perf = pROC(pred, fpr.stop=0.1)
plot(proc.perf)
abline(a=0, b= 1)

#### "optimal" cut point
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(perf, pred))

#### Base on Cutoff Values #####

# sensitivity 0.8719788
# specificity 0.9094242
# cutoff      0.1356558

pred1 = predict(m12,telecom_test,type = "response")

test_pred_churn1 <- factor(ifelse(pred1 >= 0.1356558, "Yes", "No"))

test_conf_cut <- confusionMatrix(test_pred_churn1, test_actual_churn, positive = "Yes")
test_conf_cut

# Accuracy : 0.9014
# Sensitivity : 0.8685
# Specificity : 0.9125

### KS -statistic - Test Data ######

test_cutoff_churn <- ifelse(test_pred_churn1=="Yes",1,0)
test_actual_churn <- ifelse(test_actual_churn=="Yes",1,0)


pred_object_test<- prediction(test_cutoff_churn, test_actual_churn)
performance_measures_test<- performance(pred_object_test, "tpr", "fpr")

ks_table_test <- attr(performance_measures_test, "y.values")[[1]] - 
  (attr(performance_measures_test, "x.values")[[1]])
#View(ks_table_test)
max(ks_table_test) ##### 0.7809901

# Lift & Gain Chart and Ploting ####################
# Loading dplyr package 

library(dplyr)

lift <- function(labels , predicted_prob,groups=10) {
  
  if(is.factor(labels)) labels  <- as.integer(as.character(labels ))
  if(is.factor(predicted_prob)) predicted_prob <- as.integer(as.character(predicted_prob))
  helper = data.frame(cbind(labels , predicted_prob))
  helper[,"bucket"] = ntile(-helper[,"predicted_prob"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(labels ), funs(total = n(),
                                     totalresp=sum(., na.rm = TRUE))) %>%
    
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups))) 
  return(gaintable)
}

Attrition_decile = lift(test_actual_churn, test_cutoff_churn, groups = 10)
Attrition_decile

Gain <- c(0,Attrition_decile$Gain)
Deciles <- c(0,Attrition_decile$bucket)
plot(y=Gain,x=Deciles,type ="l",lwd = 2,xlab="Bucket",ylab="Gain",main = "Gain Chart")

Random_Gain <- seq(from=0,to=100,by=10)
lines(y=Random_Gain,x=Deciles,type ="l",lwd = 2, col="red")

Perfect_Gain <- vector(mode = "numeric", length = 11)
for (i in 2:11){Perfect_Gain[i] <- 100*min(1,129*(i-1)/209)}
lines(y=Perfect_Gain,x=Deciles,type ="l",lwd = 2, col="darkgreen")


telecom_test$test_pred = predict(m12,telecom_test,type = "response")

telecom_test = telecom_test[order(telecom_test$test_pred,decreasing = T),] 

write.csv(telecom_test,"telecom_test_with_pred.csv")
##### End of Logistic Regression #########
#============================================================================================

########challenger model 1: CART
require(caret)
require(rpart)
require(rpart.plot)
require(e1071)
require(rattle)
library(gmodels)

#determining parameter value to use for CART
set.seed(100)
fitControl <- trainControl(method="CV", number=10)
cartGrid <- expand.grid(.cp=(1:50)*0.01)

#factoring the suspended variable in train and test data
telecom_training$Churn <- factor(telecom_training$Churn)
telecom_test$Churn <- factor(telecom_test$Churn)

Churn_CART <-train(Churn~.,telecom_training, method="rpart",
                   trControl=fitControl)
summary(Churn_CART)
fancyRpartPlot(Churn_CART$finalModel)


testCART <- predict(Churn_CART,telecom_test, type="raw")
confusionMatrix(telecom_test$Churn,testCART,positive = "1")


#using 10 fold CV and cp=0.01 with the rpart function

statusRpart <- rpart(Churn~.,telecom_training,method = "class",
                     control=rpart.control(cp= 0.0003914989,xval=10) )

#predicting probabilities
testRpart <- predict(statusRpart,telecom_test,type="prob")


###########################

# Let's find out the optimal probalility cutoff 
# First let's create a function to find the accuracy, sensitivity and specificity
# for a given cutoff
require(ROCR)

pred <- prediction(testRpart[,2], telecom_test$Churn)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
abline(a=0, b= 1)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]] 

#AUC = 0.9314932  Good Model

##### partial ROC curve #### 

pROC = function(pred, fpr.stop){
  perf <- performance(pred,"tpr","fpr")
  for (iperf in seq_along(perf@x.values)){
    ind = which(perf@x.values[[iperf]] <= fpr.stop)
    perf@y.values[[iperf]] = perf@y.values[[iperf]][ind]
    perf@x.values[[iperf]] = perf@x.values[[iperf]][ind]
  }
  return(perf)
}

proc.perf = pROC(pred, fpr.stop=0.1)
plot(proc.perf)
abline(a=0, b= 1)

#### "optimal" cut point
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(perf, pred))

#### Base on Cutoff Values #####

#sensitivity 0.8618858
#specificity 0.9231938
#cutoff      0.2130719
#1-0.2130719
#0.7869281


pred1 = predict(statusRpart,telecom_test,type = "prob")

test_pred_churn1 <- factor(ifelse(pred1[,2] >=  0.2130719, 1, 0))

test_conf_cut1 <- confusionMatrix(test_pred_churn1, telecom_test$Churn, positive = "1")
test_conf_cut1

#----------------------------------------------------------------

### Accuracy : 0.93 ##### 

### Exporting Data for Excel Analysis (KS, Gain, Lift etc.) ######

myeval <- matrix(nrow = length(test_pred_churn1),ncol = 2)
myeval[,1] <- test_pred_churn1
myeval[,2] <- telecom_test$Churn
colnames(myeval) <- c("Predicted_Prob","Actual_Labels")
write.csv(myeval,"myeval.csv")
#View(myeval)

### KS -statistic - Test Data ######

test_cutoff_churn <- ifelse(test_pred_churn1=="Yes",1,0)
test_actual_churn <- ifelse(telecom_test$Churn=="Yes",1,0)


pred_object_test<- prediction(test_cutoff_churn, telecom_test$Churn)

performance_measures_test<- performance(pred_object_test, "tpr", "fpr")

ks_table_test <- attr(performance_measures_test, "y.values")[[1]] - 
  (attr(performance_measures_test, "x.values")[[1]])
View(ks_table_test)
max(ks_table_test) ##### 0.5976263


# Lift & Gain Chart 

# Loading dplyr package 
library(dplyr)

lift <- function(labels , predicted_prob,groups=10) {
  
  if(is.factor(labels)) labels  <- as.integer(as.character(labels ))
  if(is.factor(predicted_prob)) predicted_prob <- as.integer(as.character(predicted_prob))
  helper = data.frame(cbind(labels , predicted_prob))
  helper[,"bucket"] = ntile(-helper[,"predicted_prob"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(labels ), funs(total = n(),
                                     totalresp=sum(., na.rm = TRUE))) %>%
    
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups))) 
  return(gaintable)
}

Attrition_decile = lift(telecom_test$Churn, test_cutoff_churn, groups = 10)
Attrition_decile

Gain <- c(0,Attrition_decile$Gain)
Deciles <- c(0,Attrition_decile$bucket)
plot(y=Gain,x=Deciles,type ="l",lwd = 2,xlab="Bucket",ylab="Gain",main = "Gain Chart")

Random_Gain <- seq(from=0,to=100,by=10)
lines(y=Random_Gain,x=Deciles,type ="l",lwd = 2, col="red")

Perfect_Gain <- vector(mode = "numeric", length = 11)
for (i in 2:11){Perfect_Gain[i] <- 100*min(1,129*(i-1)/209)}
lines(y=Perfect_Gain,x=Deciles,type ="l",lwd = 2, col="darkgreen")

##############

#ROC and AUC for Test Data
require(ROCR)
predictTest <- data.frame("Probability"=predict(statusRpart,telecom_test))
ROCRTest <- prediction(predictTest[,2],telecom_test$Churn)
ROCRTestPerf <- performance(ROCRTest,"tpr","fpr")
plot(ROCRTestPerf, colorize=T, text.adj=c(-0.2,1.7),lwd=3,
     main="ROC Curve for Predicting Churn Customers - DT model")
lines(par()$usr[1:2],par()$usr[3:4]) # 50% line for lift chart
auc <- paste(c("AUC ="),round(as.numeric(performance(ROCRTest,"auc")@y.values),digits=2),sep="")
legend("topleft",auc, bty="n")



test_pred_churn1 <- factor(ifelse(pred1 >= 0.8344524, 1, 0))

test_conf_cut <- confusionMatrix(test_pred_churn1, test_actual_churn, positive = "1")
test_conf_cut


##### WE WILL STICK TO THE LOGISTIC MODEL SINCE IT OUT PERFORMS cv CART IN TERMS OF AUC(ROC)
##### 94% VS 85%
#--------------------------------------------------------------------
library(reshape2)
library(dplyr)

fun_mean <- function(x){
  return(data.frame(y=mean(x),label=round(mean(x,na.rm=T),2)))}


#Tenure
B1<-ggplot(telecom_data,aes(x=Churn,y=tenure)) +
  geom_bar(stat = "identity",aes(fill=Churn)) +
  stat_summary(fun.y = mean, geom="point",colour="darkred", size=3) +
  stat_summary(fun.data = fun_mean, geom="text", vjust=-0.7)+
  ggtitle("Tenure Vs Churn")

#MonthlyCharges
B2<-ggplot(telecom_data,aes(x=Churn,y=MonthlyCharges)) +
  geom_bar(stat = "identity",aes(fill=Churn)) +
  stat_summary(fun.y = mean, geom="point",colour="darkred", size=3) +
  stat_summary(fun.data = fun_mean, geom="text", vjust=-0.7)+
  ggtitle("Monthly Charges Vs Churn")

#TotalCharges
B3<-ggplot(telecom_data,aes(x=Churn,y=TotalCharges)) +
  geom_bar(stat = "identity",aes(fill=Churn)) +
  stat_summary(fun.y = mean, geom="point",colour="darkred", size=3) +
  stat_summary(fun.data = fun_mean, geom="text", vjust=-0.7)+
  ggtitle("Total Charges Vs Churn")

grid.arrange(B1,B2,B3,top=textGrob("Bivariate Analysis - Churn")) 

