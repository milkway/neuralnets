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
#kaggle_test <- read_csv("~/Downloads/digit-recognizer/kaggle_test.csv")
#kaggle_train <- read_csv("~/Downloads/digit-recognizer/kaggle_train.csv")

test <- read_rds("test.rds")
label <- read_rds("label_test.rds")
train <- read_rds("train.rds")

# Visualizing a random digit
train %>% 
  select(-label, -Flag) %>% 
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
  if ("Flag" %in% colnames(base)) 
  {
    cat("\nRemoving Flag column...")
    base <- base %>%  select(-Flag)
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
Label(101:109, train)

library(keras)
set.seed(123)

#### Organize data to keras 
# train <- kaggle_train %>% 
#   mutate(Flag = sample(1:0, nrow(kaggle_train), replace = TRUE, prob = c(.10, .90)))  

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

Input <- wgts[[1]]
Input_bias <- wgts[[2]]
Layer <- wgts[[3]]
Layer_bias <- wgts[[4]]
Output <- wgts[[5]]
Output_bias <- wgts[[6]]

Y = x_test[1,] %*% Input + t(Input_bias)
Y_relu = Y * (Y > 0) # Activation Function
Z = Y_relu %*% Layer + t(Layer_bias)
Z_relu = Z * (Z > 0) # Activation Function
W = Z_relu %*% Output + t(Output_bias)
f_exp = exp(W)
W_softmax = f_exp/sum(f_exp)# softmax output

cat(round(W_softmax))

## Test image
library(imager)

digit <- load.image("digit.jpeg")
digit28 <- digit %>%  resize(size_x = 28, size_y = 28, interpolation_type = 1L)

digit28 <- rowMeans(digit28, dims = 2)

digit28 %>% apply(1, rev) %>% 
  t() %>% 
  image(col = grey.colors(256), axes = FALSE)


digit_test <- as.vector(digit28)

Y = (1 - digit_test) %*% Input + t(Input_bias)
Y_relu = Y * (Y > 0) # Activation Function
Z = Y_relu %*% Layer + t(Layer_bias)
Z_relu = Z * (Z > 0) # Activation Function
W = Z_relu %*% Output + t(Output_bias)
f_exp = exp(W)
W_softmax = f_exp/sum(f_exp)# softmax output

cat(round(W_softmax))

cat(round(W_softmax) %*% (0:9))

hist(digit_test)
