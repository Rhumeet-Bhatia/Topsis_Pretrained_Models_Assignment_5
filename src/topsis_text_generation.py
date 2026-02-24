import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw_results.csv")

models = df["model"].values
X = df.drop(columns=["model"]).values

impacts = ['-','+','-','-','+','-']

weights = np.array([0.20,0.30,0.15,0.10,0.15,0.10])

def normalize_matrix(X):
    denom = np.sqrt((X**2).sum(axis=0))
    return X/denom

R = normalize_matrix(X)
V = R * weights

A_pos = np.array([V[:,j].max() if impacts[j]=='+' else V[:,j].min() for j in range(V.shape[1])])
A_neg = np.array([V[:,j].min() if impacts[j]=='+' else V[:,j].max() for j in range(V.shape[1])])

S_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))

C = S_neg/(S_pos+S_neg)

results = pd.DataFrame({
    "Model":models,
    "TOPSIS Score":C
}).sort_values(by="TOPSIS Score",ascending=False)

print(results)

results.to_csv("../results/topsis_scores.csv",index=False)

plt.figure()
plt.bar(results["Model"],results["TOPSIS Score"])
plt.ylabel("TOPSIS Score")
plt.title("Model Ranking using TOPSIS")
plt.savefig("../results/topsis_bar_chart.png")
plt.show()