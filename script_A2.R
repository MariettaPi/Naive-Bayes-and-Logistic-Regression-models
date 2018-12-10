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

rand <- sample(nrow(mydata)/2) # Initial Permutation of
train <- mydata[rand,]
test <- mydata[-rand,]

# Specification of loss function

loss_fun01 <- function(x,y,k){
  ind <- ifelse(x==y, 0,1)
  return(sum(as.numeric(ind))/k)
}

# Naive Bayesian Model (NB)
losses_NB <- matrix(0, nrow(train), 3)
for(i in 1:nrow(train)){
  nb_model1 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 1) # train for sample size=n each time
  pred1 <- predict(nb_model1, test[seq(i),])
  nb_model2 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 0.1) # laplace = 0.1
  pred2 <- predict(nb_model2, test[seq(i),])
  nb_model3 <- naiveBayes(eval ~. ,train[seq(i),], laplace = 10) # laplace = 10
  pred3 <- predict(nb_model3, test[seq(i),])
  
  l1 <- loss_fun01(pred1, train[seq(i), 7], i)
  l2 <- loss_fun01(pred2, train[seq(i), 7], i)
  l3 <- loss_fun01(pred3, train[seq(i), 7], i)
  
  losses_NB[i,] <- c(l1,l2,l3) 
}                

head(losses_NB)


# Logistic Regression Model (LR)


LR_accuracy <-function(x,y){
  correct <- sum(as.numeric(y$eval=="Positive") * (predict(x, type="response") >= 0.5)) +
      sum(as.numeric(y$eval=="Negative") * (predict(x, type="response") < 0.5))
  tot <- sum(y$eval=="Positive") + sum(y$eval=="Negative")
  return(correct/tot)
}


losses_LR <- matrix(0, nrow(train), 2)
for(i in 4:nrow(train)){
  
  lr_model1 <- glm(eval ~. ,train[seq(i),], family = binomial) # train for sample size=n each time
  lr_model1$xlevels[["Buying"]] <- union(lr_model1$xlevels[["Buying"]], levels(test[seq(i),]$Buying))
  #pred1 <- predict(lr_model1, test[seq(i),], type = "response")
  
  lr_model2 <- glm(eval ~ Buying + Maint ,train[seq(i),], family = binomial) # laplace = 0.1
  lr_model2$xlevels[["Buying"]] <- union(lr_model2$xlevels[["Buying"]], levels(test[seq(i),]$Buying))
  #pred2 <- predict(lr_model2, test[seq(i),], type = "response")
  

  losses_LR[i,] <- c(LR_accuracy(lr_model1, test[seq(i),]), LR_accuracy(lr_model2, test[seq(i),])) 
}                

head(losses_LR)
tail(losses_LR)

# Penalized logistic regression


losses_PR <- matrix(0, nrow(train), 2)
for(i in 4:nrow(train)){
  
  pr_model1 <- glmnet(train[1:i, 7] ,train[1:i, 1:6], family = binomial) # train for sample size=n each time
  pr_model1$xlevels[["Buying"]] <- union(pr_model1$xlevels[["Buying"]], levels(test[seq(i),]$Buying))
 
  pr_model2 <- glmnet(train[1:i, 7] ,train[1:i, 1:2], family = binomial) # laplace = 0.1
  pr_model2$xlevels[["Buying"]] <- union(pr_model2$xlevels[["Buying"]], levels(test[seq(i),]$Buying))
  
  losses_PR[i,] <- c(LR_accuracy(pr_model1, test[seq(i),]), 
                     PR_accuracy(pr_model2, test[seq(i),])) 
}                

glmnet(train[1:4, 7] ,train[1:4, 1:2], family = "binomial",alpha = 0, nfolds=5) 

model.matrix( ~ .-1, train[,1:2])







#-------------------- Plot

df <- as.data.frame(losses_NB)
df_plt <- cbind(df, losses_LR, y=1:864)
rm(df)

names(df_plt)<-c("nb1", "nb2", "nb3", "lr1", "lr2", "n")

ggplot(df_plt)+
  geom_line(aes(x=log(n), y=nb1)) +
  geom_line(aes(x=log(n), y=nb2), color='red') +
  geom_line(aes(x=log(n), y=nb3), color='green') +
  geom_line(aes(x=log(n), y=lr1), color='blue') +
  geom_line(aes(x=log(n), y=lr2), color='orange') # to be modified



### Monte Carlo replicates

NB_model <- function(k, a){
  mod <- naiveBayes(eval ~. ,train[seq(k),], laplace = a)
  return(predict(mod, test[seq(k),]))
}

loss_matrix <- function(seed){
  set.seed(seed)
  train <- mydata[sample(nrow(mydata)/2),]
  test <- mydata[-sample(nrow(mydata)/2),]
  loss <- matrix(0, nrow(train), 3)
  for(i in 1:nrow(train)){
    
    pred1 <- NB_model(i, 1)
    pred2 <- NB_model(i, 0.1)
    pred3 <- NB_model(i, 10)
    
    l1 <- loss_fun01(pred1, train[seq(i), 7], i)
    l2 <- loss_fun01(pred2, train[seq(i), 7], i)
    l3 <- loss_fun01(pred3, train[seq(i), 7], i)
    
    loss[i,] <- c(l1,l2,l3) 
  }                
  return(loss)
}

head(loss_matrix(1234))


M <- list()
M <- replicate(2, loss_matrix())




