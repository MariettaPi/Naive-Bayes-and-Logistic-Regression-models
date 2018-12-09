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

rand <- sample(nrow(mydata)/2)
train <- mydata[rand,]
test <- mydata[-rand,]

# Specification of loss function

loss_fun01 <- function(x,y,k){
  ind <- ifelse(x==y, 1,0)
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
#lr <- glm(eval ~. ,train[seq(40),], family = binomial) 
#lr$xlevels[["Buying"]] <- union(lr$xlevels[["Buying"]], levels(test[seq(40),]$Buying))
#predict(lr, type = "response", newdata = subset(test[seq(40),], filt))

#-------------------- Plot

df <- as.data.frame(losses_NB)
df_plt <- cbind(df, losses_LR, y=1:864)
rm(df)

names(df_plt)<-c("nb1", "nb2", "nb3", "lr1", "lr2", "n")

ggplot(df_plt)+
  geom_line(aes(x=nb1, y=n)) +
  geom_line(aes(x=nb2, y=n)) +
  geom_line(aes(x=nb3, y=n)) +
  geom_line(aes(x=lr1, y=n)) +
  geom_line(aes(x=lr2, y=n)) # to be modified



### Monte Carlo replicates

NB_model <- function(k, a){
  mod <- naiveBayes(eval ~. ,train[seq(k),], laplace = a)
  return(predict(mod, test[seq(k),]))
}

loss_matrix <- function(){ 
  
  loss <- matrix(0, nrow(train), 3)
  for(i in 1:nrow(train)){
    train <- mydata[sample(nrow(mydata)/2),]
    test <- mydata[-sample(nrow(mydata)/2),]
    
    pred1 <- NB_model(i, 1)
    pred2 <- NB_model(i, 0.1)
    pred3 <- NB_model(i, 10)
    
    l1 <- loss_fun01(pred1, train[seq(i), 7], i)
    l2 <- loss_fun01(pred2, train[seq(i), 7], i)
    l3 <- loss_fun01(pred3, train[seq(i), 7], i)
    
    loss[i,] <- c(l1,l2,l3) 
  }                
  return(losses_NB)
}

M <- list()
M <- replicate(20, loss_matrix())



