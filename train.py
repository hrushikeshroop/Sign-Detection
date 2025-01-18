import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

data_dict = pickle.load(open('./data.pkl', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, shuffle=True, stratify=labels)

RF_model = RandomForestClassifier()
KNN_model = KNeighborsClassifier()
LR_model = LogisticRegression(max_iter=5000)

RF_model.fit(x_train, y_train)
KNN_model.fit(x_train, y_train)
LR_model.fit(x_train, y_train)

models = [RF_model, KNN_model, LR_model]
scores = []

for model in models:
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

best_model = models[np.argmax(scores)]
print(f'Best model: {best_model} with score: {np.max(scores)}')

best_model.fit(data_scaled, labels)

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': best_model}, f)

print("Model saved successfully.")
