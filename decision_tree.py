from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample dataset: Weather & Temperature
# Features: [Weather (0=Sunny, 1=Rainy), Temperature (0=Cold, 1=Hot)]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 0, 1, 1]  # Labels: (0 = No Play, 1 = Play)

# Create & Train Decision Tree Model
clf = DecisionTreeClassifier(criterion="entropy")  # Using entropy for ID3
clf.fit(X, Y)

# Visualizing the Decision Tree
plt.figure(figsize=(6,4))
plot_tree(clf, feature_names=["Weather", "Temperature"], class_names=["No", "Yes"], filled=True)
plt.show()
