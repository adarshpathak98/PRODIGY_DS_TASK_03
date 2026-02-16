import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Data load karein (Ab file upload ho chuki hai)
df = pd.read_csv('bank.csv', sep=';')

# 2. Text ko numbers mein badalna (One-hot encoding)
X = pd.get_dummies(df.drop('y', axis=1))
y = df['y']

# 3. Model training
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# 4. Diagram dikhana
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=list(X.columns), class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Prodigy InfoTech - Task 03 Decision Tree")
plt.show()
