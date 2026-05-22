import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

def load_data(path):
    return pd.read_csv(path)

def basic_stats(data):
    print("\nBasic Statistics:")
    print("Mean:", data["value"].mean())
    print("Max:", data["value"].max())
    print("Min:", data["value"].min())
    print("Standard Deviation:", data["value"].std())

def plot_data(data):
    plt.plot(data["time"], data["value"], marker="o")
    plt.title("Sensor Data Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def train_model(data):
    X = data[["value", "speed", "acceleration"]]
    y = data["label"]



    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")    
    


    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    print("Actual:", list(y_test))

    for i in range(len(X_test)):
        print("Input:", X_test.iloc[i].values)
        print("Predicted:", model.predict([X_test.iloc[i].values])[0])
        print("Actual:", y_test.iloc[i])
        print("------")
    
    new_input = pd.DataFrame([[0.5,0.8,0.5]], columns=["value", "speed", "acceleration"])
    prediction = model.predict(new_input)
    print("\nNew Input:")
    print(new_input)
    print("Prediction for new input:", prediction[0])
    


def main():
    # data = load_data(r"C:\Users\User\Desktop\sensor_Project\data\sample_data.csv")
    data = load_data("../data/sample_data.csv")
    print("Loaded Data:")
    print(data)

    basic_stats(data)
    plot_data(data)
    train_model(data)
    

if __name__ == "__main__":
    main()

