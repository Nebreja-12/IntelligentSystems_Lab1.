import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from bokeh.plotting import figure, show
from bokeh.io import output_file

df = pd.read_csv('C:\\Users\\USER\\Downloads\\Nebreja\\TB_Burden_Country.csv')
df.dropna(subset=['Estimated incidence (all forms) per 100 000 population'], inplace=True)
df['high_tb'] = (df['Estimated incidence (all forms) per 100 000 population'] > 100).astype(int)

X = df[['Country or territory name',
        'Year',
        'Estimated incidence (all forms) per 100 000 population',
        'Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population']]
y = df['high_tb']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_scores = clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)

output_file("roc_curve.html")
p = figure(title="ROC Curve", x_axis_label='False Positive Rate', y_axis_label='True Positive Rate')
p.line(fpr, tpr, line_width=2, color='navy', legend_label='ROC Curve')
p.line([0, 1], [0, 1], line_dash='dashed', color='gray', legend_label='Random Guess')
p.legend.location = "bottom_right"
show(p)