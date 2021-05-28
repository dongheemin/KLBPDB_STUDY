import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K

# region 1. 검증 function
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# endregion

dataset = pd.read_csv('./KLBPDB_HTN.csv').to_numpy()

train, test = train_test_split(dataset, test_size=0.3)

train_X = train[:, 0:16]
train_Y = train[:, 16]
test_X = test[:, 0:16]
test_Y = test[:, 16]

print(train_Y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', f1_m, precision_m, recall_m])
model.fit(train_X, train_Y, epochs=100, batch_size=10, verbose=True, validation_split=0.2)

loss, accuracy, f1_score, precision, recall = model.evaluate(test_X, test_Y)

print("\nACC : %.2f%%" % (accuracy*100))
print("F1 : %.2f%%" % (f1_score*100))
print("Pre : %.2f%%" % (precision*100))
print("Rec : %.2f%%" % (recall*100))
