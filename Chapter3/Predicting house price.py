import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models, layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 其他地方不能这么用
test_data -= mean
test_data /= std

# K-fold validation
K = 4
num_val_samples = len(train_data) // K
num_epochs = 80
all_scores = []
all_mae_histories =[]

'''
# K-fold validation

for i in range(K):
    print('processing fold #', i)
    # Prepare the validation data: data from partition #K
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

    # Prepare the training data: data from all other partition
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        epochs=num_epochs,
                        batch_size=16,
                        verbose=1)
    history_dict = history.history
    print('Keys of history:', history_dict.keys())
    mae_history = history.history['mean_absolute_error']
    all_mae_histories.append(mae_history)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plotting the average (for bigger epochs)
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

'''
# Final model
model = build_model()
# Train it on the entirety of the data
model.fit(train_data,
          train_targets,
          epochs=80,
          batch_size=16,
          verbose=1)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)