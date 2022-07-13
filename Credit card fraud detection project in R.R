# Clear the workspace
rm(list=ls())
cat("\014")
# Use menu /Session/Set Working Directory/Choose Directory Or command below to set working directory
setwd("C:/Users/sahana/Desktop")

# loading the data
cc<- read.csv("creditcard.csv", stringsAsFactors = FALSE)

# ---------------- INSTALLING LIBRARIES REQUIRED ---------------------------
# install.packages("caret")
# install.packages("rpart")
# install.packages("xgboost")
# install.packages("dplyr")
# install.packages("randomForest")
# install.packages("rpart.plot")
# install.packages("pROC")
# install.packages("stringr")
# install.packages("caTools")
# install.packages("ggplot2")
# install.packages("corrplot")
# install.packages("ROSE")
# install.packages("Rborist")
# install.packages("corrplot")
# install.packages("e1071")

library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(DMwR2) # for smote implementation
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model

# ---------------- Basic Data Exploration ---------------------------

head(cc)
str(cc)
View(cc)
names(cc)
summary(cc)
# checking missing values
colSums(is.na(cc))
# checking class imbalance
table(cc$Class)
# class imbalance in percentage
prop.table(table(cc$Class))

# ---------------- DATA VISUALIZATION ---------------------------

fig(12, 8)
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = cc, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme

#Distribution of variable 'Time' by class
cc %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y') + common_theme

#Distribution of variable 'Amount' by class
ggplot(cc, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") + common_theme

#Correlation of anonymised variables and 'Amount'
correlations <- cor(cc[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

#PIE CHART for comparing no.of frauds and non-frauds
labels = c("NON_FRAUD","FRAUD")
labels = paste(labels,round(prop.table(table(cc$Class))*100,2))
labels = paste0(labels,"%")
pie(table(cc$Class),labels,col = c("blue","red"),
    main = "Pie Chart of Credit Card Transactions")

# ---------------- Data Preparation ---------------------------

#Remove 'Time' variable
cc <- cc[,-1]
#Change 'Class' variable to factor
cc$Class <- as.factor(cc$Class)
levels(cc$Class) <- c("Not_Fraud", "Fraud")
#Scale numeric variables
cc[,-30] <- scale(cc[,-30])
head(cc)

#Split data into train and test sets
set.seed(123)
train <- sample(1:nrow(cc), nrow(cc)*0.7)
cc.train <- cc[train,]  
cc.test  <- cc[-train,]

#Choosing sampling technique
# class ratio initially
table(cc.train$Class)
# downsampling
set.seed(9560)
down_train <- downSample(x = cc.train[, -ncol(cc.train)],
                         y = cc.train$Class)
table(down_train$Class)
# upsampling
set.seed(9560)
up_train <- upSample(x = cc.train[, -ncol(cc.train)],
                     y = cc.train$Class)
table(up_train$Class)

# rose
set.seed(9560)
rose_train <- ROSE(Class ~ ., data  = cc.train)$data 
table(rose_train$Class)

# ---------------- Decision Trees (CART) with original data ---------------------------

#CART Model Performance on imbalanced data
set.seed(5627)
original_fit <- rpart(Class ~ ., data = cc.train)
original_fit 
#Confusion matrix 
pred_d <- predict(original_fit, cc.test[-30], type="class")
actual_d <- cc.test$Class
CM_d <- table(pred_d, actual_d)  
CM_d
roc.curve(cc.test$Class, pred_d, plotit = TRUE)
#AUC 0.880

####
library(rpart.plot)
prp(original_fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree")
rpart.plot(original_fit, type = 1, extra = 1, main="Classification Tree")  
rpart.plot(original_fit, type = 1, extra = 4, main="Classification Tree") 
rpart.plot(original_fit, type = 3, extra = 5, main="Classification Tree")

# ---------------- Decision Tree on various sampling techniques ---------------------------

# Build down-sampled model
set.seed(5627)
down_fit <- rpart(Class ~ ., data = down_train)
rpart.plot(down_fit, type = 3, extra = 5, main="down_fit Classification Tree")
# Build up-sampled model
set.seed(5627)
up_fit <- rpart(Class ~ ., data = up_train)
rpart.plot(up_fit, type = 3, extra = 5, main="up_fit Classification Tree")
# Build rose model
set.seed(5627)
rose_fit <- rpart(Class ~ ., data = rose_train)
rpart.plot(rose_fit, type = 3, extra = 5, main="rose_fit Classification Tree")


# AUC on down-sampled data
pred_down <- predict(down_fit, newdata = cc.test)
print('Fitting model to downsampled data')
roc.curve(cc.test$Class, pred_down[,2], plotit = TRUE)
#0.919

# AUC on up-sampled data
pred_up <- predict(up_fit, newdata = cc.test)
print('Fitting model to upsampled data')
roc.curve(cc.test$Class, pred_up[,2], plotit = TRUE)
#0.921

# AUC on up-sampled data
pred_rose <- predict(rose_fit, newdata = cc.test)
print('Fitting model to rose data')
roc.curve(cc.test$Class, pred_rose[,2], plotit = TRUE)
#0.911

# ---------------- Models on upsampled data ---------------------------

#1 Logistic Regression
library(pROC)
set.seed(42)
glm_fit <- glm(Class ~ ., data = up_train, family = 'binomial')
summary(glm_fit)
pred_glm <- predict(glm_fit, newdata = cc.test, type = 'response')
logitPredictClass <- ifelse(pred_glm > 0.5, 1, 0)
#AUC
glm.roc <- roc( cc.test$Class, pred_glm, direction="<" )
plot( glm.roc, 
      print.auc=TRUE, col = "red", lwd = 3, main = "ROC Curve for Logistical Regression" )
#Confusion matrix
CM <- table(logitPredictClass, cc.test$Class)
CM 
# PRECISON , RECALL AND F1-SCORE
TN_lg =CM[2,2]
TP_lg =CM[1,1]
FP_lg =CM[1,2]
FN_lg =CM[2,1]
precision_lg =(TP_lg)/(TP_lg+FP_lg)
recall_score_lg =(TP_lg)/(FN_lg+TP_lg)
f1_score_lg=2*((precision_lg*recall_score_lg)/(precision_lg+recall_score_lg))
accuracy_model_lg  =(TP_lg+TN_lg)/(TP_lg+TN_lg+FP_lg+FN_lg)
False_positive_rate_lg =(FP_lg)/(FP_lg+TN_lg)
False_negative_rate_lg =(FN_lg)/(FN_lg+TP_lg)
print(CM)
print(accuracy_model_lg)
print(precision_lg)
print(recall_score_lg)
print(f1_score_lg)
#0.980

#2 Random Forest
library(randomForest)
# build random forest model using every variable
set.seed(42)
rfModel <- randomForest(Class ~ . , data = up_train, ntree =30)
cc.test$predicted <- predict(rfModel, cc.test[,-30])
#Confusion matrix
cm_rf=confusionMatrix(cc.test$Class, cc.test$predicted)
cm_rf
#AUC
roc.curve(cc.test$Class, cc.test$predicted, plotit = TRUE)
#0.886
# PRECISON , RECALL AND F1-SCORE
TN_rf =119
TP_rf =85283
FP_rf =6
FN_rf =35
precision_rf =(TP_rf)/(TP_rf+FP_rf)
recall_score_rf =(TP_rf)/(FN_rf+TP_rf)
f1_score_rf=2*((precision_rf*recall_score_rf)/(precision_rf+recall_score_rf))
accuracy_model_rf  =(TP_rf+TN_rf)/(TP_rf+TN_rf+FP_rf+FN_rf)
False_positive_rate_rf =(FP_rf)/(FP_rf+TN_rf)
False_negative_rate_rf =(FN_rf)/(FN_rf+TP_rf)
print(cm_rf)
print(accuracy_model_rf)
print(precision_rf)
print(recall_score_rf)
print(f1_score_rf)

# 3. Naive Bayes Classifier ########### 
#install.packages("e1071")
library(e1071)
# run naive bayes
fit.nb <- naiveBayes(Class ~ ., data = up_train)
fit.nb
# Evaluate Performance using Confusion Matrix
pred <- predict(fit.nb, cc.test)
actual=cc.test$Class
nbPredictClass <- predict(fit.nb, cc.test, type = "class")
#AUC
roc.curve(cc.test$Class, nbPredictClass, plotit = TRUE)
#cm
cm_nb <- table(nbPredictClass, actual)
cm_nb 
# PRECISON , RECALL AND F1-SCORE
TN_nb =cm_nb[2,2]
TP_nb =cm_nb[1,1]
FP_nb =cm_nb[1,2]
FN_nb =cm_nb[2,1]
precision_nb =(TP_nb)/(TP_nb+FP_nb)
recall_score_nb =(TP_nb)/(FN_nb+TP_nb)
f1_score_nb=2*((precision_nb*recall_score_nb)/(precision_nb+recall_score_nb))
accuracy_model_nb  =(TP_nb+TN_nb)/(TP_nb+TN_nb+FP_nb+FN_nb)
False_positive_rate_nb =(FP_nb)/(FP_nb+TN_nb)
False_negative_rate_nb =(FN_nb)/(FN_nb+TP_nb)
print(cm_nb)
print(accuracy_model_nb)
print(precision_nb)
print(recall_score_nb)
print(f1_score_nb)
#0.907

#4 XGBoost
# Convert class labels from factor to numeric
library(xgboost)
labels <- up_train$Class
y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)
set.seed(42)
xgb <- xgboost(data = data.matrix(up_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.4,
               max_depth = 4, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.5,
               verbose = 0,
               nthread = 10,)
xgb_pred <- predict(xgb, data.matrix(cc.test[,-30]))
#AUC
roc.curve(cc.test$Class, xgb_pred, plotit = TRUE)
#0.983

#AUC
xgb.roc <- roc( cc.test$Class, xgb_pred, direction="<" )
plot( xgb.roc, 
      print.auc=TRUE, col = "blue", lwd = 3, main = "ROC Curve for XGBoost " )
#Confusion matrix
xgb.threshold <- 0.4
CM_xg <- table(as.numeric( xgb_pred > xgb.threshold ), cc.test$Class)
CM_xg 
#0.983
# PRECISON , RECALL AND F1-SCORE
TN_xg =CM_xg[2,2]
TP_xg =CM_xg[1,1]
FP_xg =CM_xg[1,2]
FN_xg =CM_xg[2,1]
precision_xg =(TP_xg)/(TP_xg+FP_xg)
recall_score_xg =(TP_xg)/(FN_xg+TP_xg)
f1_score_xg=2*((precision_xg*recall_score_xg)/(precision_xg+recall_score_xg))
accuracy_model_xg  =(TP_xg+TN_xg)/(TP_xg+TN_xg+FP_xg+FN_xg)
False_positive_rate_xg =(FP_xg)/(FP_xg+TN_xg)
False_negative_rate_xg =(FN_xg)/(FN_xg+TP_xg)
print(CM_xg)
print(precision_xg)
print(recall_score_xg)
print(f1_score_xg)

# ---------------- look at the important features ---------------------------
names <- dimnames(data.matrix(up_train[,-30]))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# graph
xgb.plot.importance(importance_matrix[1:8,])

# ---------------- THE END ----------------------------------------------------