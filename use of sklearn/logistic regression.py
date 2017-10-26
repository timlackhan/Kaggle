from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    #import the data set
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    #split the data set into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    #generate a lr classifier model
    classifier = LogisticRegression(penalty='l2',dual=False,tol=0.0001, C=1.0, fit_intercept=True)

    #train the model
    classifier.fit(X_train, y_train)
    print(classifier)

    #give the probabilities of the test set
    print(classifier.predict_proba(X_test))

    #give the predictive label of the test set
    print(classifier.predict(X_test))