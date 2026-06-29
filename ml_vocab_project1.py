"""
Train a tiny classifier using scikit learn
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_classifier():
    """Train the iris plants toy dataset and train it to classify items"""
    data = load_iris() 

    # extract the x(features) and y(label)
    X, y = data['data'], data['target'] 

    # Split the data into 66 training and 33 test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

    # Train the model (from scratch using our data)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    

if __name__ == "__main__":
    train_classifier()