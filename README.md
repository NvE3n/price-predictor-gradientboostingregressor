# price-predictor-gradientboostingregressor

This is a simple machine learning model that predicts laptop prices by inserting specifications. Mainly used pre-built linear models, ensemble models and some other reggression models. 

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost.sklearn import XGBRegressor
```

## Used machiene learning model accuracies

> [!NOTE]
> Below model accuracies will be changed.

### Linear Models
```
ElasticNet     - 0.6177668613600819
SGDRegressor     - 0.5981049488462731
LinearRegression - 0.7368696612581807
Lasso            - 0.7368144300281375
BayesianRidge    - 0.7390254720176629
```

### Ensemble Models
```
RandomForestRegressor         - 0.7931818772343384
GradientBoostingRegressor     - 0.8012627085647115
HistGradientBoostingRegressor - 0.7936760380366881
```

### Other Models
```
SVR          - 0.023262447195445478
KernelRidge  - 0.7393279723774349
XGBRegressor - 0.7909195707044037
```

## Hyperparameter-tuning

- The ***GradientBoostingRegressor*** model got the highest accuracy. Used it for further model development.

Used parameters :
```python
param_grid = {'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
              'learning_rate':[0.0, 0.01, 0.1, 0.2, 0.3],
              'n_estimators': [200, 300, 400]}
```

### Best Estimator :
```
GradientBoostingRegressor(loss='huber', n_estimators=300)
```
#### Final accuracy : 0.8428347881268542 

Special Thanks for @Dinesh S Piyasamara :shipit: 
