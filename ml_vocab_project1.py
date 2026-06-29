"""
Train a tiny classifier using scikit learn
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_classifier():
    """Train the iris plants toy dataset and train it to classify items"""
    data = load_iris() 

    # extract the x(features) and y(label)
    X, y = data['data'], data['target'] 

    # Split the data into 66 training and 33 test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

    # Train the model (from scratch using our data)
    model = LogisticRegression(max_iter=300)

    # Give the model dataset with labeled examples to learn patterns
    model.fit(X_train, y_train)

    # What will the model predict given unlabeled features(x)
    model_predictions = model.predict(X_test)

    # Evaluate how well the model fares(1 perfect, 0 average, - worst)
    score = accuracy_score(y_true=y_test, y_pred=model_predictions, normalize=True)
    print(f"The accuracy score for this model is: {score}")

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_classifier()
    
    # Test on real test data
    prediction = model.predict(X_test[10].reshape(1, -1))
    print("Actual label:", y_test[10])
    print("Predicted label:", prediction[0])