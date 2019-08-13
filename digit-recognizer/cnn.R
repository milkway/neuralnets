

# CNN

# Redefine dimension of train/test inputs
x_train_cnn <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test_cnn <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))

#image(x_train_cnn[3,,,1])

model_cnn <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
  ) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  # these are the embeddings (activations) we are going to visualize
  layer_dense(units = 128, activation = 'relu', name = 'features') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

summary(model_cnn)

history <- model_cnn %>% fit(
  x_train_cnn, y_train, 
  epochs = 12, batch_size = 128, 
  validation_split = 0.2
)


# Evaluate
model_cnn %>% evaluate(x_test_cnn, y_test)
