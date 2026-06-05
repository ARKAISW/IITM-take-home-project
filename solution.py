# personal finance risk analysis
import pandas as pd
import numpy as np

d = pd.read_csv('Dataset.csv')

# basic stats
print("Avg income:", d['MonthlyIncome'].mean())
print(d['FinancialRisk'].value_counts())
print(d.groupby('FinancialRisk')['SavingsRate'].mean())

# correlation with risk
r = {'Low':0, 'Medium':1, 'High':2}
d['r'] = d['FinancialRisk'].map(r)
for c in ['Age','MonthlyIncome','MonthlyExpenses','SavingsRate','DebtToIncomeRatio','NumCreditCards']:
    print(f"{c}: {d[c].corr(d['r']):.3f}")

print(d.groupby('FinancialRisk')['DebtToIncomeRatio'].mean())
print("Medium %:", d[d['FinancialRisk']=='Medium'].shape[0]/200*100)
print(d.groupby('FinancialRisk')['NumCreditCards'].mean())
print(d.groupby('FinancialRisk')['MonthlyIncome'].mean())
print("Savings vs DTI:", d['SavingsRate'].corr(d['DebtToIncomeRatio']))

# centroid calc
print("Centroid:", np.mean([20,24,28,32]))

# distance
p = np.array([30, 0.15])
a = np.array([12, 0.48])
b = np.array([32, 0.18])
print("dist A:", np.linalg.norm(p-a), "dist B:", np.linalg.norm(p-b))

print("Cluster size:", 10-3+5)

# gini
print("Left:", len(d[d['DebtToIncomeRatio']<0.35]))
print("Right:", len(d[d['DebtToIncomeRatio']>=0.35]))

g1 = 1-(77/78)**2-(1/78)**2
g2 = 1-(3/122)**2-(69/122)**2-(50/122)**2
print("Gini left:", round(g1,3))
print("Gini right:", round(g2,3))
print("Split %:", 72/200*100)
print("Equal gini:", round(1-3*(1/3)**2, 3))
print("P(medium):", round(18/80, 3))

d.drop(columns='r', inplace=True)
