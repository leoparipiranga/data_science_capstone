################################
# MovieLens Project Report
################################
#
#
#by Jose Leonardo Ribeiro Nascimento
#
#
#
#
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#
#Information about datasets
#

#This function returns dimensions (number of rows and number of columns) of a data.frame.

dim(edx)

dim(validation)


#Head returns the first part of an object.

head(edx)

head(validation)

#This code returns how many unique users and unique movies are in edx set.

edx %>%
  summarise(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))


#This code returns a matrix for a random sample of 100 movies and 100 users, with an image with yellow indicating a user movie combination for which we have a rating.

users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")


#This code groups edx dataset by movieId and returns a graph showing the distribution of movie rates by movie.

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")


#This code groups edx dataset by userId and returns a graph showing the distribution of movie rates by user.

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")



#
#############################
#CREATING TRAIN AND TEST SET
#############################
#

#In order to test the predictors, I've decided to split the EDX set into train_set and test_set, using the following code:


#First, I must load caret package, which includes the tools for data splitting.

library(caret)


#Then I set the seed to 1:

set.seed(1, sample.kind="Rounding")

#Finally, I split the edx set, with 80% of the data to train_set and 20% for test_set:

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#To make sure we don’t include users and movies in the test_set that do not appear in the training set, I removed these using the semi_join function:

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")



#
#CREATING RMSE FUNCTION
#
#


#This function computes the residual mean squared error for a vector of ratings and their corresponding predictors.

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



#
#
#FIRST TRY: ONLY THE AVERAGE
#

#This function returns the average rating of all movies across all users in the train set.

mu_hat <- mean(train_set$rating)
mu_hat


#This function returns the RMSE of the first model applied on the test set data.

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse 

#This function creates a table to store the results as I improve my model.

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)


#
#
#FIRST MODEL: MOVIE EFFECT
#
#

#fit <- lm(rating ~ as.factor(MovieId), data = train_set)
#Can't use this function because the dataset is too large


#This code first calculates average rating for every movie and then calculates b_i, which is the average of Y_u_i minus the overall mean for each movie

mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))


#This code plots the b_i effect.

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#This code applies the model with the b_i effect to test set

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

#Calculates the RMSE for the model

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

#Add the result of the new model to the rmse_results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()



#
#
#
#SECOND MODEL: ADDING USER EFFECT
#
#

#This code computes the average rating for users who have rated over than 100 movies and make a histogram of those values

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")


#This code calculates b_u, which is the average of Y_u_i minus the overall mean for each movie, minus b_i


user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#This code applies the model with the b_u effect to test set

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#Calculates the RMSE for the model

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

#Add the result of the new model to the rmse_results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()


#
#
#
#ADDING REGULARIZATION TO THE FIRST TWO MODELS
#


#This code creates a simple table with movieId and title, without repetition

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

#Shows the top 10 movies ordered by rating, according to the first model, which only considers movie effects

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

#Shows the worst 10 movies ordered by rating, according to the first model, which only considers movie effects

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

#This code returns the top ten movies, but adding number of reviews for each movie

train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


#This code returns the worst ten movies, but adding number of reviews for each movie

train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#This code apply regularization to the first model, considering lambda value as 3, which is already the optimized value

lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

#This code plots original estimates vs regularized estimates, showing that as ni as small, values are shrinking more towards zero

data_frame(original = movie_avgs$b_i, 
           regularized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#Top 10 movies using regularization 

train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


#Worst 10 movies using regularization 

train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#
#
#
#APPLYING REGULARIZATION TO THE FIRST MODEL
#


#This code applies the model with the b_i effect and regularization to test set

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

#Calculates the RMSE for the model

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)

#Add the result of the new model to the rmse_results table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()


#
#
#
#
#APPLYING REGULARIZATION TO THE SECOND MODEL
#


#This code apply regularization to the second model (user effect), using cross-validation to choose the best value for lambda

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})


#Plot lambda and rmses value to demonstrate the best value for lambda

qplot(lambdas, rmses)  

#Shows the lambda which generates minimum RMSE 

lambda <- lambdas[which.min(rmses)]
lambda

#Add the result of the new model to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#
#
#CREATING THIRD MODEL: ADDING THE GENRE EFFECT
#
#
#
#

#This code shows how many unique genres combination are present in edx dataset

n_distinct(edx$genres)

#This code calculates b_g, which is the average of Y_u_i minus the overall mean for each movie, minus b_i, minus b_u


genre_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))



#This code applies the model with the b_g effect to test set

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId")%>%
  left_join(user_avgs, by = "userId")%>%
  left_join(genre_avgs, by = "genres")%>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
           .$pred
         
#Calculates the RMSE for the model
         
model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
         
#Add the result of the new model to the rmse_results table
         
rmse_results <- bind_rows(rmse_results,
             data_frame(method="Movie + User + Genre Effects Model",  
              RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()
         

#
#
#REGULARIZING THIRD MODEL
#
#


#This code apply regularization to the third model (movie + user + genres effect), using cross-validation to choose the best value for lambda

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u- mu)/(n()+l))
  
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

#Plot lambda and rmses value to demonstrate the best value for lambda

qplot(lambdas, rmses)  

#Shows the lambda which generates minimum RMSE 

lambda <- lambdas[which.min(rmses)]
lambda

#Add the result of the new model to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genre Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#
#
#
#FOURTH MODEL: ADDING YEAR EFFECT
#
#
#


#Load lubridate package (install if needed with #install.packages(“lubridate”)

library(lubridate)

#Add date and year columns to train and test set using lubridate package command as_datetime

train_set<-mutate(train_set, date = as_datetime(timestamp),year=year(date))

test_set<-mutate(test_set, date = as_datetime(timestamp),year=year(date))


#This code calculates b_y, which is the average of Y_u_i minus the overall mean for each movie, minus b_i, minus b_u, minus b_g


year_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))


#This code applies the model with the b_y effect to test set

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  .$pred

#Calculates the RMSE for the model

model_5_rmse <- RMSE(predicted_ratings, test_set$rating)

#Add the result of the new model to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + Genre + Year Effects Model",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()

#
#
#APPLYING REGULARIZATION TO FOURTH MODEL
#
#


#This code apply regularization to the third model (movie + user + genres + year effect), using cross-validation to choose the best value for lambda

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u- mu)/(n()+l))
  b_y <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u- b_g)/(n()+l))
  
  
  
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})


#Plot lambda and rmses value to demonstrate the best value for lambda

qplot(lambdas, rmses)  

#Shows the lambda which generates minimum RMSE 

lambda <- lambdas[which.min(rmses)]
lambda

#Add the result of the new model to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genre + Year Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()



#####################################################
#
#FINAL STEP: APPLYING FINAL MODEL TO VALIDATION SET
#
#####################################################

#Add date and year columns to edx and validation set using lubridate package command as_datetime

edx<-mutate(edx, date = as_datetime(timestamp),year=year(date))

validation<-mutate(validation, date = as_datetime(timestamp),year=year(date))

#And now I apply my final model, considering lambda (l=5):

#This code applies the final model (Regularized movie + user + genres + year effect) to edx and validation set
  
l<-5

mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u- mu)/(n()+l))

b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u- b_g)/(n()+l))    


predicted_ratings <-
  test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  .$pred


# Calculate the predicted values for the validation data set

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)


# Final RMSE
Final_RMSE <- RMSE(predicted_ratings, validation$rating)
Final_RMSE


###########################################################
#
#FINAL RMSE OBTAINED WITH THIS SCRIPT WAS 0.8644101 
#
###########################################################
