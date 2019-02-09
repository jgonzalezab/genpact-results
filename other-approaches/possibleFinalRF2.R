# Libraries
library(caret) # library(earth)
library(Metrics)

############################### FIRST APPROACH ############################### 
# Data
train <- read.csv('Data/train.csv')
trainCenter <- read.csv('Data/fulfilment_center_info.csv')
trainMeal <- read.csv('Data/meal_info.csv')
test <- read.csv('Data/test_QoiMO9B.csv')
submission <- read.csv('Data/sample_submission_hSlSoT6.csv')

# Data engineering
train$id <- NULL # Delete id
train$emailer_for_promotion <- factor(train$emailer_for_promotion) # Factorize
train$homepage_featured <- factor(train$homepage_featured) # Factorize

test$emailer_for_promotion <- factor(test$emailer_for_promotion) # Factorize
test$homepage_featured <- factor(test$homepage_featured) # Factorize

trainCenter$city_code <- factor(trainCenter$city_code) # Factorize
trainCenter$region_code <- factor(trainCenter$region_code) # Factorize

# Joining datasets by center_id
trainDEF <- merge(train, trainCenter, by = 'center_id')
trainDEF$center_id <- factor(trainDEF$center_id) # Factorize

testDEF <- merge(test, trainCenter, by = 'center_id')
testDEF$center_id <- factor(testDEF$center_id) # Factorize

# Reordering columns
trainDEF <- trainDEF[, c(1:7, 9:ncol(trainDEF), 8)]
testDEF <- testDEF[, c(2, 1, 3:ncol(testDEF))]

# 1. Spliting train by meal
# 2. Split into train and valid for each meal
#    For each meal we select the las 10 weeks as test and the rest as train
trainBYmeal <- list()
for(meal in unique(trainDEF$meal_id)) {
  
  auxDF <- trainDEF[trainDEF$meal_id == meal, -3] # Delete meal_id column
  
  trainBYmeal[[as.character(meal)]][['train']] <- auxDF[auxDF$week %in% 1:135, ]
  trainBYmeal[[as.character(meal)]][['test']] <- auxDF[auxDF$week %in% 136:145, ]
  
}

# Define loss function
RMSLECaretFunc <- function(data, lev = NULL, model = NULL) {
  
  rmsleCaret <- Metrics::rmsle(data$obs, data$pred)
  c(RMSLEcaret = -rmsleCaret)
  
}

# CSV with results
# Submission_1: Contains predictions for the test and predictions reduced by 0.9
# and predictions augmented by 1.10
submission_1 <- submission
submission_1 <- cbind.data.frame(submission_1,
                                 num_ordersReduced = rep(0, nrow(submission_1)),
                                 num_ordersAugmented = rep(0, nrow(submission_1)))

# The corresponding metrics for the test set extracted from the training
metrics <- data.frame(meal = rep(0, length(trainBYmeal)),
                      RMSLENormal = rep(0, length(trainBYmeal)),
                      RMSLEreduced = rep(0, length(trainBYmeal)),
                      RMSLEAugmented = rep(0, length(trainBYmeal)))

# Train the models on each meal id.
# First find optimal hyperparameters by validation and calculate RMSLE on test
# Once the optimal hyperparameters are known fit the model on all the train data
# Lastly makes the predictions for that meal_id
# We will fit EARTH Models
set.seed(15122018)
ctrl <- trainControl(method = 'cv',
                     number = 5,
                     summaryFunction = RMSLECaretFunc)

cat('\014') # Clean screen

for(i in 1:length(trainBYmeal)) {
  
  cat(i, ' | 51', sep = '') # Progress info
  cat('\n')
  
  elemMeal <- trainBYmeal[[i]]
  
  # Find optimal paramaters
  model <- train(num_orders ~ .,
                 data = elemMeal[['train']],
                 method = 'earth',
                 trControl = ctrl,
                 tuneLength = 10,
                 metric = 'RMSLEcaret')
  
  optmParam <- model$bestTune
  
  # Predictions over the pseudo test set
  predsPseudo <- predict(model, elemMeal[['test']][, -11])
  
  # Update metrics CSV
  metrics[i, 1] <- names(trainBYmeal[i])
  metrics[i, 2] <- Metrics::rmsle(predsPseudo, elemMeal[['test']][, 11])
  metrics[i, 3] <- Metrics::rmsle(predsPseudo, elemMeal[['test']][, 11] * 0.9)
  metrics[i, 4] <- Metrics::rmsle(predsPseudo, elemMeal[['test']][, 11] * 1.1)
  
  # Fit final model
  modelFinal <- train(num_orders ~ .,
                      data = rbind.data.frame(elemMeal[['train']],
                                              elemMeal[['test']]),
                      method = 'earth',
                      trControl = trainControl(method = 'none',
                                               summaryFunction = RMSLECaretFunc),
                      tuneGrid = optmParam,
                      metric = 'RMSLEcaret')
  
  # Final predictions over the real test set
  testAUX <- testDEF[testDEF$meal_id == as.numeric(names(trainBYmeal[i])),
                     -4]
  predsFinal <- predict(modelFinal, testAUX[, -1])
  testAUX$preds <- predsFinal
  
  for (j in 1:nrow(testAUX)) {
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_orders'] <-
      testAUX[j, 'preds']
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_ordersReduced'] <-
      testAUX[j, 'preds'] * 0.9
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_ordersAugmented'] <-
      testAUX[j, 'preds'] * 1.1
    
  } 
  
}

write.csv(submission_1, file = 'submission_1.csv', row.names = FALSE)
write.csv(metrics, file = 'metrics.csv', row.names = FALSE)

# Once generated these files we can make submisons based on:
# Original pred
# Reduced pred
# Augmented pred
# Combination of them
metrics <- read.csv('metrics.csv')
submission_1 <- read.csv('submission_1.csv')

# Original pred
preds_1_sub <- submission_1[, c(1, 2)]
write.csv(preds_1_sub, 'preds_1_sub.csv', row.names = FALSE) 

# Reduced pred
preds_2_sub <- submission_1[, c(1, 3)]
colnames(preds_2_sub)[2] <- 'num_orders'
write.csv(preds_2_sub, 'preds_2_sub.csv', row.names = FALSE)

# Augmented pred
preds_3_sub <- submission_1[, c(1, 4)]
colnames(preds_3_sub)[2] <- 'num_orders'
write.csv(preds_3_sub, 'preds_3_sub.csv', row.names = FALSE)

# Combined pred (test dataframe needed)
preds_4_sub <- submission_1[, c(1, 2)]
preds_4_sub[, 2] <- rep(0, nrow(preds_4_sub))

metricsMins <- data.frame(meal = metrics[, 1],
                          whichMin = apply(metrics[, -1], 1, which.min))

for(i in preds_4_sub$id) {
  
  mealFORid <- test[test$id == i, 'meal_id']
  bestPred <- metricsMins[metricsMins$meal == mealFORid, 2]
  
  preds_4_sub[preds_4_sub$id == i, 'num_orders'] <-
    submission_1[submission_1$id == i, bestPred + 1]
  
}

write.csv(preds_4_sub, 'preds_4_sub.csv', row.names = FALSE) # LB: 58.3519
##############################################################################
cat('\014')
############################### SECOND APPROACH ##############################
# Data
train <- read.csv('Data/train.csv')
trainCenter <- read.csv('Data/fulfilment_center_info.csv')
trainMeal <- read.csv('Data/meal_info.csv')
test <- read.csv('Data/test_QoiMO9B.csv')
submission <- read.csv('Data/sample_submission_hSlSoT6.csv')

# Data engineering
train$id <- NULL # Delete id
train$emailer_for_promotion <- factor(train$emailer_for_promotion) # Factorize
train$homepage_featured <- factor(train$homepage_featured) # Factorize

test$emailer_for_promotion <- factor(test$emailer_for_promotion) # Factorize
test$homepage_featured <- factor(test$homepage_featured) # Factorize

# Joining datasets by meal_id
trainDEF <- merge(train, trainMeal, by = 'meal_id')
trainDEF$meal_id <- factor(trainDEF$meal_id) # Factorize

testDEF <- merge(test, trainMeal, by = 'meal_id')
testDEF$meal_id <- factor(testDEF$meal_id) # Factorize

# Reordering columns
trainDEF <- trainDEF[, c(1:7, 9:ncol(trainDEF), 8)]
testDEF <- testDEF[, c(2, 1, 3:ncol(testDEF))]

# 1. Spliting train by center
# 2. Split into train and valid for each center
#    For each center we select the las 10 weeks as test and the rest as train
trainBYcenter <- list()
for(center in unique(trainDEF$center_id)) {
  
  auxDF <- trainDEF[trainDEF$center_id == center, -3] # Delete center_id column
  
  trainBYcenter[[as.character(center)]][['train']] <- auxDF[auxDF$week %in% 1:135, ]
  trainBYcenter[[as.character(center)]][['test']] <- auxDF[auxDF$week %in% 136:145, ]
  
}

# Define loss function
RMSLECaretFunc <- function(data, lev = NULL, model = NULL) {
  
  rmsleCaret <- Metrics::rmsle(data$obs, data$pred)
  c(RMSLEcaret = -rmsleCaret)
  
}

# CSV with results
# Submission_1: Contains predictions for the test and predictions reduced by 0.9
# and predictions augmented by 1.10
submission_1 <- submission
submission_1 <- cbind.data.frame(submission_1,
                                 num_ordersReduced = rep(0, nrow(submission_1)),
                                 num_ordersAugmented = rep(0, nrow(submission_1)))

# The corresponding metrics for the test set extracted from the training
metrics <- data.frame(center = rep(0, length(trainBYcenter)),
                      RMSLENormal = rep(0, length(trainBYcenter)),
                      RMSLEreduced = rep(0, length(trainBYcenter)),
                      RMSLEAugmented = rep(0, length(trainBYcenter)))

# Train the models on each center id.
# First find optimal hyperparameters by validation and calculate RMSLE on test
# Once the optimal hyperparameters are known fit the model on all the train data
# Lastly makes the predictions for that meal_id
# We will fit EARTH Models
set.seed(15122018)
ctrl <- trainControl(method = 'cv',
                     number = 5,
                     summaryFunction = RMSLECaretFunc)

cat('\014') # Clean screen

for(i in 1:length(trainBYcenter)) {
  
  cat(i, ' | 77', sep = '') # Progress info
  cat('\n')
  
  elemCenter <- trainBYcenter[[i]]
  
  # Find optimal paramaters
  model <- train(num_orders ~ .,
                 data = elemCenter[['train']],
                 method = 'earth',
                 trControl = ctrl,
                 tuneLength = 10,
                 metric = 'RMSLEcaret')
  
  optmParam <- model$bestTune
  
  # Predictions over the pseudo test set
  predsPseudo <- predict(model, elemCenter[['test']][, -11])
  
  # Update metrics CSV
  metrics[i, 1] <- names(trainBYcenter[i])
  metrics[i, 2] <- Metrics::rmsle(predsPseudo, elemCenter[['test']][, 'num_orders'])
  metrics[i, 3] <- Metrics::rmsle(predsPseudo, elemCenter[['test']][, 'num_orders'] * 0.9)
  metrics[i, 4] <- Metrics::rmsle(predsPseudo, elemCenter[['test']][, 'num_orders'] * 1.1)
  
  # Fit final model
  modelFinal <- train(num_orders ~ .,
                      data = rbind.data.frame(elemCenter[['train']],
                                              elemCenter[['test']]),
                      method = 'earth',
                      trControl = trainControl(method = 'none',
                                               summaryFunction = RMSLECaretFunc),
                      tuneGrid = optmParam,
                      metric = 'RMSLEcaret')
  
  # Final predictions over the real test set
  testAUX <- testDEF[testDEF$center_id == as.numeric(names(trainBYcenter[i])),
                     -4]
  predsFinal <- predict(modelFinal, testAUX[, -1])
  testAUX$preds <- predsFinal
  
  for (j in 1:nrow(testAUX)) {
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_orders'] <-
      testAUX[j, 'preds']
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_ordersReduced'] <-
      testAUX[j, 'preds'] * 0.9
    
    submission_1[submission_1$id == testAUX[j, 'id'], 'num_ordersAugmented'] <-
      testAUX[j, 'preds'] * 1.1
    
  } 
  
}

write.csv(submission_1, file = 'submission_1_2nd.csv', row.names = FALSE)
write.csv(metrics, file = 'metrics_2nd.csv', row.names = FALSE)

# Once generated these files we can make submisons based on:
# Original pred
# Reduced pred
# Augmented pred
# Combination of them
metrics <- read.csv('metrics_2nd.csv')
submission_1 <- read.csv('submission_1_2nd.csv')

# Original pred
preds_1_sub <- submission_1[, c(1, 2)]
write.csv(preds_1_sub, 'preds_1_sub_2nd.csv', row.names = FALSE) 

# Reduced pred
preds_2_sub <- submission_1[, c(1, 3)]
colnames(preds_2_sub)[2] <- 'num_orders'
write.csv(preds_2_sub, 'preds_2_sub_2nd.csv', row.names = FALSE) 

# Augmented pred
preds_3_sub <- submission_1[, c(1, 4)]
colnames(preds_3_sub)[2] <- 'num_orders'
write.csv(preds_3_sub, 'preds_3_sub_2nd.csv', row.names = FALSE) 

# Combined pred (test dataframe needed)
preds_4_sub <- submission_1[, c(1, 2)]
preds_4_sub[, 2] <- rep(0, nrow(preds_4_sub))

metricsMins <- data.frame(center = metrics[, 1],
                          whichMin = apply(metrics[, -1], 1, which.min))

for(i in preds_4_sub$id) {
  
  centerFORid <- test[test$id == i, 'center_id']
  bestPred <- metricsMins[metricsMins$center == centerFORid, 2]
  
  preds_4_sub[preds_4_sub$id == i, 'num_orders'] <-
    submission_1[submission_1$id == i, bestPred + 1]
  
}

write.csv(preds_4_sub, 'preds_4_sub_2nd.csv', row.names = FALSE) 
##############################################################################

################################## ENSEMBLE ##################################
firstApproachPREDS <- read.csv('preds_2_sub.csv') # Reduced preds (first approach)
secondApproachPREDS <- read.csv('preds_2_sub_2nd.csv') # Reduced preds (second approach)

ensemble <- data.frame(id = firstApproachPREDS$id,
                       num_orders = rowMeans(data.frame(firstApproachPREDS$num_orders,
                                                        secondApproachPREDS$num_orders)))

write.csv(ensemble, file = 'ensembleFS.csv', row.names = FALSE)
##############################################################################

