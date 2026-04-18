import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    df = pd.read_csv("logs/scores.csv")

    plt.plot(df["episode"], df["score"], marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Heuristic Agent Learning Curve")
    plt.grid(True)
    
    # Save the plot explicitly as well
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/learning_curve.png")
    
    plt.show()
    print("Plot saved to logs/learning_curve.png")
