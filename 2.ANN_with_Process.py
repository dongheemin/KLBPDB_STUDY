import tensorflow as tf
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

# region 2. Variable Selection PipeLine Fuction
def pipeline_logit_kbest(X_train, Y_train):
    select = SelectKBest(score_func=f_classif)

    scaler = StandardScaler()
    logit_model = LogisticRegression()

    pipe = Pipeline([('scaler', scaler), ('feature_selection', select), ('model', logit_model)])

    param_grid = [{'feature_selection__k': [3, 5, 7],
                   'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'model__penalty': ['l1', 'l2']
                   }]

    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    return grid_search
# endregion

dataset = pd.read_csv('./KLBPDB_HTN.csv')

X_train = dataset.iloc[:, 0:16]
Y_train = dataset.iloc[:, 16]

grid_search_kbest = pipeline_logit_kbest(X_train, Y_train)

mask = grid_search_kbest.best_estimator_.named_steps['feature_selection'].get_support()
features_list = list(X_train.columns.values)

selected_features = []
for bool, features in zip(mask, features_list):
    if bool:
        selected_features.append(features)

renew_dataset = pd.DataFrame()
for cols in selected_features:
    renew_dataset.loc[:,str(cols)] = dataset[cols]
renew_dataset.loc[:,'outcomes'] = dataset['HP_DX_YN']
train_renew, test_renew = train_test_split(renew_dataset.to_numpy(), test_size=0.3)

train_X = train_renew[:, 0:7]
train_Y = train_renew[:, 7]
test_X = test_renew[:, 0:7]
test_Y = test_renew[:, 7]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=7),
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
