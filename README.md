# Plotly library AI and ML Functions

Plotly, it is a powerful data visualization library that supports a wide range of chart types and customization options. It provides interactive and visually appealing plots that can be embedded in web applications.

## What About Dash?
Dash is an open-source framework for building analytical applications, with no Javascript required, and it is tightly integrated with the Plotly graphing library.

Learn how to install Dash:(https://dash.plotly.com/installation)

## Artificial Intelligence and Machine Learning

### Plotly Python Open Source Graphing Library AI and ML Charts
Plotly.py is free and open source. Plotly's Python graphing library makes interactive, publication-quality graphs related to AI and ML:
1) ML Regression in Python
2) kNN Classification
3) ROC and PR Curves
4) PCA Visualization
5) AI/ML APps with Dash
6) t-SNE and UMAP Projections

## 1) ML Regression in Python
Visualize regression in scikit-learn with Plotly. Plotly charts can be used for displaying various types of regression models with various capabilities, such as comparative analysis of the same model with different parameters, displaying Latex, surface plots for 3D data, and enhanced prediction error analysis with Plotly Express.

We will use Scikit-learn to split and preprocess our data and train various regression models. Scikit-learn is a popular Machine Learning (ML) library that offers various tools for creating and training ML algorithms, feature engineering, data cleaning, and evaluating and testing models. It was designed to be accessible, and to work seamlessly with popular libraries like NumPy and Pandas.

### ML Regression in Dash
Dash is the best way to build analytical apps in Python using Plotly figures. 
To run the app below, run `pip install dash`, click "Download" to get the code and run `python app.py`.
``` Python
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

app = Dash(__name__)

models = {'Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}

app.layout = html.Div([
    html.H4("Predicting restaurant's revenue"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["Regression", "Decision Tree", "k-NN"],
        value='Decision Tree',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input('dropdown', "value"))
def train_and_display(name):
    df = px.data.tips() # replace with your own data source
    X = df.total_bill.values[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, df.tip, random_state=42)

    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    ])
    return fig

app.run_server(debug=True)
```
### Model generalization on unseen data
With `go.Scatter`, you can easily color your plot based on a predefined data split. By coloring the training and the testing data points with different colors, you can easily see if whether the model generalizes well to the test data or not.

```Python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = px.data.tips()
X = df.total_bill[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, df.tip, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))


fig = go.Figure([
    go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
    go.Scatter(x=x_range, y=y_range, name='prediction')
])
fig.show()
```


### Comparing different kNN models parameters
In addition to linear regression, it's possible to fit the same data using k-Nearest Neighbors.When you perform a prediction on a new sample, this model either takes the weighted or un-weighted average of the neighbors. In order to see the difference between those two averaging options, we train a kNN model with both of those parameters, and we plot them in the same way as the previous graph.
```Python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)
x_range = np.linspace(X.min(), X.max(), 100)

# Model #1
knn_dist = KNeighborsRegressor(10, weights='distance')
knn_dist.fit(X, df.tip)
y_dist = knn_dist.predict(x_range.reshape(-1, 1))

# Model #2
knn_uni = KNeighborsRegressor(10, weights='uniform')
knn_uni.fit(X, df.tip)
y_uni = knn_uni.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', color='sex', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_uni, name='Weights: Uniform'))
fig.add_traces(go.Scatter(x=x_range, y=y_dist, name='Weights: Distance'))
fig.show()
```


## 2) kNN Classification in Python

Visualize scikit-learn's k-Nearest Neighbors (kNN) classification in Python with Plotly.

We will train a k-Nearest Neighbors (kNN) classifier. First, the model records the label of each training sample. Then, whenever we give it a new sample, it will look at the k closest samples from the training set to find the most common label, and assign it to our new sample. We will use Scikit-learn for training our model and for loading and splitting data.

This section gets us started with displaying basic binary classification using 2D data. We first show how to display training versus testing data using various marker styles, then demonstrate how to evaluate our classifier's performance on the test split using a continuous color gradient to indicate the model's predicted score.

### Display training and test splits

Using Scikit-learn, we first generate synthetic data that form the shape of a moon. We then split it into a training and testing set. Finally, we display the ground truth labels using a scatter plot.
In the graph, we display all the negative labels as squares, and positive labels as circles. We differentiate the training and test set by adding a dot to the center of test data.

In this example, we will use graph objects, Plotly's low-level API for building figures.

```Python
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load and split data
X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y.astype(str), test_size=0.25, random_state=0)

trace_specs = [
    [X_train, y_train, '0', 'Train', 'square'],
    [X_train, y_train, '1', 'Train', 'circle'],
    [X_test, y_test, '0', 'Test', 'square-dot'],
    [X_test, y_test, '1', 'Test', 'circle-dot']
]

fig = go.Figure(data=[
    go.Scatter(
        x=X[y==label, 0], y=X[y==label, 1],
        name=f'{split} Split, Label {label}',
        mode='markers', marker_symbol=marker
    )
    for X, y, label, split, marker in trace_specs
])
fig.update_traces(
    marker_size=12, marker_line_width=1.5,
    marker_color="lightyellow"
)
fig.show()
```
## 3) ROC and PR Curves
Interpret the results of your classification using Receiver Operating Characteristics (ROC) and Precision-Recall (PR) Curves in Python with Plotly.

Basic binary ROC curve
```Python
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, random_state=0)

model = LogisticRegression()
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()
```

We also display the area under the ROC curve (ROC AUC), which is fairly high, thus consistent with our interpretation of the previous plots. 
```Python
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn import metrics, datasets
import plotly.express as px

app = Dash(__name__)

MODELS = {'Logistic Regression': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}

app.layout = html.Div([
    html.H4("Analysis of the ML model's results using ROC and PR curves"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["Logistic Regression", "Decision Tree", "k-NN"],
        value='Logistic Regression',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input('dropdown', "value"))
def train_and_display(model_name):
    X, y = datasets.make_classification( # replace with your own data source
        n_samples=1500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return fig


app.run_server(debug=True)
```

### ROC curve in Dash
Dash is the best way to build analytical apps in Python using Plotly figures. To run the app below, run pip install dash, click "Download" to get the code and run python app.py.

```Python
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn import metrics, datasets
import plotly.express as px

app = Dash(__name__)

MODELS = {'Logistic Regression': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}

app.layout = html.Div([
    html.H4("Analysis of the ML model's results using ROC and PR curves"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["Logistic Regression", "Decision Tree", "k-NN"],
        value='Logistic Regression',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input('dropdown', "value"))
def train_and_display(model_name):
    X, y = datasets.make_classification( # replace with your own data source
        n_samples=1500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return fig


app.run_server(debug=True)
```
## 4) PCA Visualization
## 5) AI/ML APps with Dash
## 6) t-SNE and UMAP Projections

## Plot CSV Data in Python

### How to create charts from csv files with Plotly and Python
```Python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv')
df.head()
```

### Plot from CSV with Plotly Express
``` Python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv')

fig = px.line(df, x = 'AAPL_x', y = 'AAPL_y', title='Apple Share Prices over time (2014)')
fig.show()
```

### Plot from CSV in Dash
Dash is the best way to build analytical apps in Python using Plotly figures. 
To run the app below, run `pip install dash`, click "Download" to get the code and run `python app.py`.

``` Python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Simple stock plot with adjustable axis'),
    html.Button("Switch Axis", n_clicks=0, 
                id='button'),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input("button", "n_clicks"))
def display_graph(n_clicks):
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv') # replace with your own data source

    if n_clicks % 2 == 0:
        x, y = 'AAPL_x', 'AAPL_y'
    else:
        x, y = 'AAPL_y', 'AAPL_x'

    fig = px.line(df, x=x, y=y)    
    return fig


app.run_server(debug=True)
```


