##################################################
# ECON 418-518 Homework 3
# Ananya Singhvi
# The University of Arizona
# ananyasinghvi@arizona.edu 
# 9 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(ISLR2, glmnet, boot, data.table, ggplot2, readxl)

# Set sead
set.seed(418518)


#####################
# Problem 1
#####################

# Loading the data
dt <- read.csv("/Users/hps/Desktop/Econometrics/HW3.csv")

# Converting to data.table
dt <- as.data.table(dt)

#################
# Question (i)
#################

#Dropping Specified Columns from the data
dt <- dt[, !c("fnlwgt", "occupation", "relationship", "capital.gain", "capital.loss", "educational.num"), with = FALSE]

##############
# Question (ii)
##############

##############
# Part (a)
##############

# Convert income to a binary indicator. if > 50,000, then 1, otherwise 0
dt[, income := ifelse(income == ">50K",1 ,0)] 

##############
# Part (b)
##############

# Convert race to binary indicator. If White, then 1, otherwise 0
dt[, race := ifelse(race == "White", 1, 0)]

##############
# Part (c)
##############

# Convert gender to binary indicator. If Male, then 1, otherwise 0
dt[, gender := ifelse(gender == "Male", 1, 0)]

##############
# Part (d)
##############

# Convert workclass to binary indicator. If Private, then 1, otherwise 0
dt[, workclass := ifelse(workclass == "Private", 1, 0)]

##############
# Part (e)
##############

# Convert native.country to binary indicator. If United-States, then 1, otherwise 0
dt[, native.country := ifelse(native.country == "United-States", 1, 0)]

##############
# Part (f)
##############

# Convert marital.status to binary indicator. If Married-civ-spouse, then 1, otherwise 0
dt[, marital.status := ifelse(marital.status == "Married-civ-spouse", 1, 0)]

##############
# Part (g)
##############
# Convert education to binary indicator. If Masters/ Bachelors/ Doctorate, then 1, otherwise 0
dt[, education := ifelse(education == "Bachelors" | education == "Masters"
                         | education == "Doctorate", 1, 0)]

##############
# Part (h)
##############

# Age Squared Variable
dt[, age_sq := age^2]

##############
# Part (i)
##############

# Standardize the age, age_sq, and hours per week variables (x - mean(x)/ sd(x))
dt[,':='(
  age_std = (age - mean(age)) / (sd(age)),
  age_sq_std = (age_sq - mean(age_sq)) / (sd(age_sq)),
  hours.per.week_std = 
    (hours.per.week - mean(hours.per.week)) / (sd(hours.per.week))
)]

##############
# Question (iii)
##############

##############
# Part (a)
##############

# Calculate the proportion of individuals with income greater than 50K
mean(dt$income)

##############
# Part (b)
##############

# Calculate the proportion of individuals in the private sector
mean(dt$workclass)

##############
# Part (c)
##############

# Calculate the proportion of married individuals
mean(dt$marital.status)

##############
# Part (d)
##############

# Calculate the proportion of females in the dataset
mean(dt$gender == 0)

##############
# Part (e)
##############

# Calculate the total number of NAs in the dataset
sum(is.na(dt))

##############
# Part (f)
##############

# Convert the "income" variable to a factor data type
dt[, income := as.factor(income)]

#################
# Question (iv)
#################

##############
# Part (a)
##############

#last training set observation value
train.size <- floor(0.7 * nrow(dt))
print(train.size)

##############
# Part (b)
##############
# Create the training data table 
shuffle.index <- sample(nrow(dt))
train.index <- shuffle.index[1:train.size]
train.dt <- dt[train.index, ]

##############
# Part (c)
##############
# Create the test data table
test.size <- ceiling(0.3 * nrow(dt))
test.size
test.index <- shuffle.index[(train.size + 1):nrow(dt)]
test.dt <- dt[test.index, ]

#################
# Question (v)
#################

##############
# Part (b)
##############

#Loading Caret Package
library(caret)

# Prepare the feature set with X and Y variables
X <- model.matrix(income ~ ., train.dt)[, -1]
y <- train.dt[, income]  

# Create a sequence of 50 lambda values
lambda_grid <- 10^seq(5, -2, length.out = 50)

# Train the lasso regression model using caret's train() function
lasso_model <- train(
  x = X, 
  y = as.factor(y), 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10), 
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid) 
)

# View the best lambda value and accuracy
lasso_model$bestTune$lambda
max(lasso_model$results$Accuracy)

##############
# Part (d)
##############

# Variables with coefficient estimates that are approximately 0
coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

##############
# Part (e)

# Extract coefficients 
lasso_coefs <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)

# Convert the S4 object to a matrix
lasso_coefs_matrix <- as.matrix(lasso_coefs)

# Identify non-zero coefficients
nonzero_coefs <- rownames(lasso_coefs_matrix)[lasso_coefs_matrix[, 1] != 0]

# Remove the intercept
nonzero_vars <- setdiff(nonzero_coefs, "(Intercept)")

# Filter training and testing datasets to include non-zero variables
training_filtered <- train.dt[, c(nonzero_vars, "income"), with = FALSE]
testing_filtered <- test.dt[, c(nonzero_vars, "income"), with = FALSE]

# Create feature matrix
Xfiltered_train <- model.matrix(income ~ ., training_filtered)[, -1]

# Create outcome vector
yfiltered_train <- training_filtered[, income]

# Define a grid of lambda values
grid <- 10^seq(5, -2, length = 50)

# Train Lasso model
lasso_model <- train(
  x = Xfiltered_train, 
  y = as.factor(yfiltered_train), 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = grid)
)

# Print results
print(lasso_model)

# View the best lambda value and accuracy
lasso_model$bestTune$lambda
max(lasso_model$results$Accuracy)

# Train Ridge model
ridge_model <- train(
  x = Xfiltered_train, 
  y = as.factor(yfiltered_train), 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = grid)  # Set alpha = 0 for Ridge regression
)

# Print results
print(ridge_model)

# View the best lambda value and accuracy
ridge_model$bestTune$lambda
max(ridge_model$results$Accuracy)

#################
# Question (vi)
#################

##############
# Part (b)
##############
pacman::p_load(caret, randomForest)

X <- model.matrix(~ . - income, data = dt)[, -1]
y <- as.factor(dt$income)
ntree_values <- c(100, 200, 300)
mtry_values <- c(2, 5, 9)
rf_model <- train(
  x = X, 
  y = y, 
  method = "rf",
  trControl = trainControl(method = "cv", number = 5), 
  tuneGrid = expand.grid(mtry = mtry_values),  # Number of random features to split
  ntree = 300  # Train models with 300 trees (for speed) but will adjust below
)
# Train Random Forests with 100, 200, and 300 trees
rf_model_100 <- randomForest(x = X, y = y, mtry = 5, ntree = 100, importance = TRUE)
rf_model_200 <- randomForest(x = X, y = y, mtry = 5, ntree = 200, importance = TRUE)
rf_model_300 <- randomForest(x = X, y = y, mtry = 5, ntree = 300, importance = TRUE)

# Print the model summaries
print(rf_model_100)
print(rf_model_200)
print(rf_model_300)
