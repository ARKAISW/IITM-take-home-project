# generate all figures for the report
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

os.makedirs('figures', exist_ok=True)
d = pd.read_csv('Dataset.csv')
C = ['#2ecc71','#f39c12','#e74c3c']
R = ['Low','Medium','High']

# risk distribution
rc = d['FinancialRisk'].value_counts()
fig,ax = plt.subplots()
bars = ax.bar(R, [rc[r] for r in R], color=C, edgecolor='k', lw=.5)
for b,v in zip(bars,[rc[r] for r in R]): ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontweight='bold')
ax.set(xlabel='Financial Risk', ylabel='Count', title='Distribution of Financial Risk')
plt.tight_layout(); plt.savefig('figures/risk_distribution.png', dpi=150); plt.close()

# scatter savings vs dti
fig,ax = plt.subplots()
cm = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}
for r in R:
    s = d[d['FinancialRisk']==r]
    ax.scatter(s['SavingsRate'], s['DebtToIncomeRatio'], c=cm[r], label=r, alpha=.7, edgecolors='k', lw=.3)
ax.set(xlabel='Savings Rate (%)', ylabel='DTI', title='Savings Rate vs DTI'); ax.legend()
plt.tight_layout(); plt.savefig('figures/scatter_savings_dti.png', dpi=150); plt.close()

# correlation heatmap
fig,ax = plt.subplots(figsize=(8,6))
cols = ['Age','MonthlyIncome','MonthlyExpenses','SavingsRate','DebtToIncomeRatio','NumCreditCards']
sns.heatmap(d[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=.5)
ax.set_title('Correlation Heatmap')
plt.tight_layout(); plt.savefig('figures/correlation_heatmap.png', dpi=150); plt.close()

# grouped bar charts
fig,axes = plt.subplots(2,2,figsize=(10,8))
for ax,m in zip(axes.flat, ['MonthlyIncome','SavingsRate','DebtToIncomeRatio','NumCreditCards']):
    d.groupby('FinancialRisk')[m].mean().reindex(R).plot(kind='bar', ax=ax, color=C, edgecolor='k', lw=.5)
    ax.set(title=f'Avg {m}', xlabel=''); ax.set_xticklabels(R, rotation=0)
plt.suptitle('Key Metrics by Risk', fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('figures/grouped_metrics.png', dpi=150); plt.close()

# boxplots
fig,axes = plt.subplots(1,3,figsize=(14,5))
for ax,c in zip(axes, ['MonthlyIncome','SavingsRate','DebtToIncomeRatio']):
    d.boxplot(column=c, by='FinancialRisk', ax=ax, positions=[0,1,2])
    ax.set(title=c, xlabel='Risk'); ax.set_xticklabels(R)
plt.suptitle('Box Plots by Risk', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('figures/boxplots.png', dpi=150); plt.close()

# elbow method
dk = d.drop(columns=['PersonID','FinancialRisk'])
dk['Gender'] = LabelEncoder().fit_transform(dk['Gender'])
X = MinMaxScaler().fit_transform(dk)
wcss = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in range(1,11)]
fig,ax = plt.subplots()
ax.plot(range(1,11), wcss, 'bo-', lw=2, ms=8)
ax.axvline(3, color='r', ls='--', label='K=3')
ax.set(xlabel='K', ylabel='WCSS', title='Elbow Method'); ax.legend(); ax.set_xticks(range(1,11))
plt.tight_layout(); plt.savefig('figures/elbow_plot.png', dpi=150); plt.close()

# kmeans clusters
km = KMeans(3, n_init=10, random_state=42).fit(X)
d['Cluster'] = km.labels_
fig,ax = plt.subplots()
cc = ['#3498db','#e74c3c','#2ecc71']
for c in sorted(d['Cluster'].unique()):
    s = d[d['Cluster']==c]
    ax.scatter(s['SavingsRate'], s['DebtToIncomeRatio'], c=cc[c], label=f'C{c}', alpha=.7, edgecolors='k', lw=.3)
ax.set(xlabel='Savings Rate (%)', ylabel='DTI', title='K-Means Clusters (K=3)'); ax.legend()
plt.tight_layout(); plt.savefig('figures/kmeans_clusters.png', dpi=150); plt.close()

# decision tree
dd = d.drop(columns=['PersonID','Cluster'])
dd['Gender'] = LabelEncoder().fit_transform(dd['Gender'])
Xd, y = dd.drop(columns='FinancialRisk'), dd['FinancialRisk']
Xtr,Xte,ytr,yte = train_test_split(Xd, y, test_size=.2, random_state=42, stratify=y)
dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42).fit(Xtr, ytr)

fig,ax = plt.subplots(figsize=(20,10))
plot_tree(dt, feature_names=list(Xd.columns), class_names=['High','Low','Medium'], filled=True, rounded=True, ax=ax, fontsize=8, proportion=True)
ax.set_title('Decision Tree (depth=4)', fontsize=14)
plt.tight_layout(); plt.savefig('figures/decision_tree.png', dpi=150); plt.close()

# feature importance
imp = pd.Series(dt.feature_importances_, index=Xd.columns).sort_values(ascending=False)
fig,ax = plt.subplots()
imp.plot(kind='barh', ax=ax, color='#3498db', edgecolor='k'); ax.invert_yaxis()
ax.set(xlabel='Importance', title='Feature Importance')
plt.tight_layout(); plt.savefig('figures/feature_importance.png', dpi=150); plt.close()

# confusion matrix
fig,ax = plt.subplots(figsize=(6,5))
cm2 = confusion_matrix(yte, dt.predict(Xte), labels=['Low','Medium','High'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=R, yticklabels=R, ax=ax)
ax.set(xlabel='Predicted', ylabel='Actual', title='Confusion Matrix')
plt.tight_layout(); plt.savefig('figures/confusion_matrix.png', dpi=150); plt.close()

print("done")
