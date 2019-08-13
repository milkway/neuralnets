# Digit Recognizer in R 
# Using Neura Networks 

# Loading Libraries
library(tidyverse)
library(keras)
#install_keras()

# Constants
img_rows <- 28
img_cols <- 28
n_classes <- 10

# Loading Data
kaggle_test <- read_csv("~/Downloads/digit-recognizer/kaggle_test.csv")
kaggle_train <- read_csv("~/Downloads/digit-recognizer/kaggle_train.csv")


# Visualizing a random digit

kaggle_test %>% 
  sample_n(1) %>%   
  unlist() %>%
  matrix(nrow = 28, byrow = TRUE) %>% 
  apply(2, rev) %>% 
  t() %>% 
  image() 

# Creating a function 

# Start Function
Label <- function(digits, base, nrows = 28, ncols = 28){
  stopifnot(digits %in% 1:nrow(base))
  if ("label" %in% colnames(base)) 
  {
    cat("\nRemoving label column...")
    base <- base %>%  select(-label)
  }
  # Check graphical parameters 
  val <- par(no.readonly=TRUE)
  n <- ceiling(sqrt(length(digits)))
  par(mfrow = c(ceiling(length(digits)/n),n), mar = c(0.1, 0.1, 0.1, 0.1))
  for (i in digits){ 
    m <- base %>% 
      filter(row_number() == i) %>%  
      unlist() %>%
      matrix(nrow = nrows, ncol = ncols, byrow = TRUE) %>% 
      apply(2, rev) %>% 
      t() %>% 
    image(col = grey.colors(255), axes = FALSE)
  }
  # reset the original graphics parameters
  par(val)                                               
}

# Call Function
Label(1:10, kaggle_train)

library(keras)
#### Organize data to keras 
train <- kaggle_train %>% 
  mutate(Flag = sample(1:0, nrow(kaggle_train), replace = TRUE, prob = c(.10, .90)))  

x_train <- train %>% 
  filter(Flag == 0) %>% 
  select(-label, -Flag) %>% 
  mutate_all(function(x) x/255) %>% 
  as.matrix()

x_test  <- train %>% 
  filter(Flag == 1) %>% 
  select(-label, -Flag) %>% 
  mutate_all(function(x) x/255) %>% 
  as.matrix()

y_train <- train %>% 
  filter(Flag == 0) %>% 
  select(label)  %>% as.matrix() %>% 
  to_categorical(num_classes = 10)

y_test <- train %>% 
  filter(Flag == 1) %>% 
  select(label)  %>% 
  as.matrix() %>% 
  to_categorical(num_classes = 10)

# Defining the RNN Model
rnn_model <- keras_model_sequential() 
rnn_model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(img_cols*img_rows)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = n_classes, activation = 'softmax')

summary(rnn_model)

rnn_model %>% compile(
  loss = 'categorical_crossentropy', #categorical_accuracy
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Training and Evaluation

history <- rnn_model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)   

# Evaluate
rnn_model %>% evaluate(x_test, y_test)


# Predict
rnn_model %>% predict_classes(x_test)
