# Import dataset
rm(list = ls())
library(e1071)
library(ggplot2)
library(glmnet)
options(scipen = "999")

cardata <- read.table("~/Documents/Statistical Learning/cardata_eval.txt", sep = ",")
mydata <- data.frame(cardata[, 1:6], eval = ifelse(cardata[, 7] == "unacc", "Negative",
                                               "Positive"))

names(mydata) <- c("Buying", "Maint", "Doors", "Persons", "Lug_Boot", "safety", "eval")

rand <- sample(nrow(mydata), size = nrow(mydata)/2) # Initial Permutation of rows
train <- mydata[rand,]
test <- mydata[-rand,]

# Specification of loss function

loss_fun01 <- function(x,y){
  ind <- ifelse(x==y, 0,1)
  return(sum(ind)/length(y))
}

# Naive Bayesian Model (NB)
losses_NB <- matrix(0, nrow(train), 3)
for(i in 1:nrow(train)){
  nb_model1 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 1) # train for sample size=n each time
  pred1 <- predict(nb_model1, test)
  nb_model2 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 0.1) # laplace = 0.1
  pred2 <- predict(nb_model2, test)
  nb_model3 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 10) # laplace = 10
  pred3 <- predict(nb_model3, test)
  
  l1 <- loss_fun01(pred1, test[, 7])
  l2 <- loss_fun01(pred2, test[, 7])
  l3 <- loss_fun01(pred3, test[, 7])
  
  losses_NB[i,] <- c(l1,l2,l3) 
}                

head(losses_NB)

nb_1 <- naiveBayes(eval ~. ,train[seq(1),], laplace = 1) # train for sample size=n each time
pred1 <- predict(nb_1, test)

loss_fun01(pred1, test[,7])


# Logistic Regression Model (LR)


#LR_acc <-function(x,y){
#  correct <- sum(as.numeric(y$eval=="Positive") * (predict(x, type="response") >= 0.5)) +
#      sum(as.numeric(y$eval=="Negative") * (predict(x, type="response") < 0.5))
#  tot <- sum(y$eval=="Positive") + sum(y$eval=="Negative")
#  return(1-correct/tot)
#}

LR_error <- function(x,y){
  pred <- predict(x, y[,-7] ,type="response")
  thres <- pred >= 0.5 
  pred[thres] <- "Positive"
  pred[!thres] <- "Negative"
  ind <- ifelse(pred==y[,7], 0, 1)
  return((sum(ind)/nrow(y)))
} 



losses_LR <- matrix(0, nrow(train), 2)
for(i in 15:nrow(train)){
  
  lr_model1 <- glm(eval ~. ,train[seq(i),], family = binomial("logit")) # train for sample size=n each time
  #lr_model1$xlevels[["Buying"]] <- union(lr_model1$xlevels[["Buying"]], levels(test$Buying))
  #pred1 <- predict(lr_model1, test[seq(i),], type = "response")
  
  lr_model2 <- glm(eval ~ Buying + Maint ,train[seq(i),], family = binomial) # laplace = 0.1
  #lr_model2$xlevels[["Buying"]] <- union(lr_model2$xlevels[["Buying"]], levels(test$Buying))
  #pred2 <- predict(lr_model2, test[seq(i),], type = "response")
  

  losses_LR[i,] <- c(LR_error(lr_model1, test), 
                     LR_error(lr_model2, test)) 
}                

head(losses_LR)
tail(losses_LR)

# Penalized logistic regression

temp <- model.matrix(eval~ ., data=mydata) [,-1]
Xmat <- as.matrix(data.frame(temp))

pn_train <- Xmat[rand,]
pn_test <- Xmat[-rand,]


PR_error <- function(x, y, dt){
  pred <- predict(x, y ,type="response", s="lambda.min")
  thres <- pred >= 0.5 
  pred[thres] <- "Positive"
  pred[!thres] <- "Negative"
  ind <- ifelse(pred==dt[,7], 0, 1)
  return((sum(ind)/nrow(dt)))
}

losses_PR <- matrix(0, nrow(train), 2)
for(i in 24:nrow(train)){
  pr_model1 <- cv.glmnet(pn_train[1:i,] ,y = train[1:i,7], family = "binomial", alpha=0) # train for sample size=n each time
  tm_pred <- predict.cv.glmnet(pr_model1, pn_test, s="lambda.min"
                               ,type="response") 
  
  pr_model2 <- cv.glmnet(pn_train[1:i,1:6] ,y = train[1:i,7], family = "binomial", alpha=0)
  tm_pred <- predict.cv.glmnet(pr_model2, pn_test, s="lambda.min"
                               ,type="response") 
  
  losses_PR[i,] <- c(PR_error(pr_model1, pn_test, test), 
                     PR_error(pr_model2, pn_test, test)) 
}                





mod <-  cv.glmnet(pn_train[1:24,1:6] ,y = train[1:24,7], family = "binomial", alpha=0)
tm_pred <- predict.cv.glmnet(mod, pn_test, s="lambda.min",type="response")

PR_error(mod, pn_test, test)


#-------------------- Plot

df <- as.data.frame(losses_NB)
df_plt <- cbind(df, losses_LR, y=1:864)
rm(df)

names(df_plt)<-c("nb1", "nb2", "nb3", "lr1", "lr2", "n")

ggplot(df_plt)+
  geom_line(aes(x=n, y=nb1)) +
  geom_line(aes(x=n, y=nb2), color='red') +
  geom_line(aes(x=n, y=nb3), color='green') +
  geom_line(aes(x=n, y=lr1), color='blue') +
  geom_line(aes(x=n, y=lr2), color='orange') # to be modified



### Monte Carlo replicates

NB_model <- function(dt, tst, k, a){
  mod <- naiveBayes(eval ~. ,dt[seq(k),], laplace = a)
  return(predict(mod, tst[seq(k),]))
}

set.seed(1234)
loss_matrix <- function(){
  
  ind<-sample(nrow(mydata), size = nrow(mydata)/2)
  train <- mydata[ind,]
  test <- mydata[-ind,]
  loss <- matrix(0, nrow(train), 5)
  for(i in 34:nrow(train)){
    
    pred1 <- NB_model(train, test, i, 1)
    pred2 <- NB_model(train, test, i, 0.1)
    pred3 <- NB_model(train, test, i, 10)
    
    lr_mod1 <- glm(eval ~. ,train[seq(i),], family = binomial) 
    
    lr_mod2 <- glm(eval ~ Buying + Maint ,train[seq(i),], family = binomial) 
    
    l1 <- loss_fun01(pred1, test[, 7])
    l2 <- loss_fun01(pred2, test[, 7])
    l3 <- loss_fun01(pred3, test[, 7])
    
    loss[i,] <- c(l1, l2, l3, LR_error(lr_mod1, test), LR_error(lr_mod2, test)) 
  }                
  return(loss)
}

#head(loss_matrix(1234))


M <- list()
for(i in 1:20){
  M[[i]] <- loss_matrix()  
}

#1:04
(M[[1]] + M[[2]])/2

saveRDS(M, file = "mymat.Rds")
