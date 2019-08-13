
library(neuralnet)

## Example

# Binary classification
nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris, linear.output = FALSE)
print(nn)
plot(nn)

##


# Set up formula

x_train_nn <- train %>% 
  filter(Flag == 0) %>% 
  select(-label, -Flag) %>% 
  mutate_all(function(x) x/255) 

x_test_nn  <- train %>% 
  filter(Flag == 1) %>% 
  select(-label, -Flag) %>% 
  mutate_all(function(x) x/255)

y_train_nn <- train %>% 
  filter(Flag == 0) %>% 
  select(label)  %>%  as.matrix() %>%   
  to_categorical(num_classes = 10) %>% 
  as_tibble()

y_test_nn <- train %>% 
  filter(Flag == 1) %>% 
  select(label)  %>% 
  as.matrix() %>% 
  to_categorical(num_classes = 10) %>% 
  as_tibble()

train_nn <- bind_cols(y_train_nn, x_train_nn)
# Formula
f <- as.formula(paste(paste(names(y_train_nn), collapse = " + "),
                      "~", 
                      paste(names(x_train_nn), collapse = " + ")))


nn <- neuralnet(f,
                data = train_nn,
                hidden = c(128, 10),
                act.fct = "logistic",
                linear.output = FALSE,
                lifesign = "minimal")


# Plot neural net
plot(nn)

# Compute predictions
probs.nn <- compute(nn, x_test_nn)

# Extract results
probs.nn_ <- probs.nn$net.result
head(probs.nn_)

# Accuracy (training set)
predicted_nn <- max.col(probs.nn_)
target <- max.col(y_test_nn)

head(predicted_nn, n = 10)
head(target, n = 10)

mean(predicted_nn == target)

