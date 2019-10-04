from sklearn import datasets
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)

class ScrapyKNN():
    def fit(self, X_train, Y_train ):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]


iris = datasets.load_iris(return_X_y=True)

X = iris.data()
Y = iris.target()


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.5)
my_classifier = ScrapyKNN()
my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)

print (accuracy_score(Y_test, predictions))
