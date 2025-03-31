import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_random_forest():
    df = pd.read_csv("preprocessed_data.csv")
    df["quality"] = (df["temperature"] < 30).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["quality"]), df["quality"], test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
