#Predicting Medical Costs Using Machine Learning In R

# Loading necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ggcorrplot)
library(car)
library(corrplot)
library(MASS)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lm.beta)

# Setting working directory and loading the dataset
setwd("C:/Users/LDO SYSTEMS/Desktop/R Data Sci")
data <- read.csv("insurance.csv")
head(data)
View(data)

# Checking the structure and summary of the data
str(data)
summary(data)

#checking for missing values
colSums(is.na(data))

# Replacing zero values in specific columns with NA
cols_to_fix <- c("children")
data[cols_to_fix] <- lapply(data[cols_to_fix], function(x) ifelse(x == 0, NA, x))

# Imputing missing values using median
data[cols_to_fix] <- lapply(data[cols_to_fix], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
view(data)

# checking for outliers
boxplot(data$charges)
boxplot(data$bmi)
boxplot(data$age)

# checking the distribution of charges using a histogram
ggplot(data, aes(x = charges)) +
  geom_histogram(binwidth = 500, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Charges", x = "Charges", y = "Count")

# Selecting only numerical columns
numerical_data <- data[, sapply(data, is.numeric)]

# Computing correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs") 
print(correlation_matrix)

#converting categorical variables into factor
data[c("sex", "smoker", "region")] <- lapply(data[c("sex", "smoker", "region")], as.factor)

#scaling numerical varaibles 
data <- data %>%
  mutate(across(where(is.numeric), scale))

# Checking Variance Inflation Factor (VIF)
vif_model <- lm(charges ~ ., data = data)
vif_values <- vif(vif_model)
print(vif_values)

# Splitting data into training (80%) and testing (20%) sets
set.seed(123)
trainIndex <- createDataPartition(data$charges, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Building regression model
lm_model <- lm(charges ~ ., data = trainData)
summary(lm_model)  # Check model summary

# Building decision tree model
tree_model <- rpart(charges ~ ., data = trainData, method = "anova")
rpart.plot(tree_model, type = 2, fallen.leaves = TRUE)

# Predicting on regression model 
lm_pred <- predict(rm, newdata = testData)

# Predicting on decision tree model 
tree_pred <- predict(tree_model, newdata = testData)

## Computing Performance Metrics

# Defining a Function for Metrics
evaluate_model <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  r2 <- cor(actual, predicted)^2
  
  return(list(MSE = mse, RMSE = rmse, MAE = mae, R2 = r2))
}

# Evaluate Linear Regression
lm_metrics <- evaluate_model(testData$charges, lm_pred)
print("Linear Regression Metrics:")
print(lm_metrics)

# Evaluate Decision Tree
tree_metrics <- evaluate_model(testData$charges, tree_pred)
print("Decision Tree Metrics:")
print(tree_metrics)

## Creating scatter plot to compare actual and predicted values

# Store actual values
actual_values <- testData$charges

# Compare Actual vs. Linear Regression Predictions
ggplot(data = data.frame(actual_values, lm_pred), aes(x = actual_values, y = lm_pred)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted (Linear Regression)",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

# Compare Actual vs. Decision Tree Predictions
ggplot(data = data.frame(actual_values, tree_pred), aes(x = actual_values, y = tree_pred)) +
  geom_point(color = "green", alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted (Decision Tree)",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

# Identifying key factors that significantly impact charges
lm_beta <- lm.beta(lm_model)  # Standardized coefficients
print(lm_beta)

rpart.plot(tree_model)  # Visualize the tree


