import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tpot import TPOTClassifier
from imblearn.over_sampling import SMOTE

class MLPredictor:
    def __init__(self, data_base: pd.DataFrame, features: list, n_lags:int):
        self.data_base = data_base
        self.features = features
        self.n_lags = n_lags
        self.data_base = self.create_lag_features(self.data_base, self.n_lags)
        self.features = [f'{feature}_lag_{n_lags}' for feature in self.features]
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def create_lag_features(self, data, n_lags=1):
        """Create lag features for time series data."""
        lagged_data = data.copy()
        for lag in range(1, n_lags + 1):
            for feature in self.features:
                lagged_data[f'{feature}_lag_{lag}'] = lagged_data[feature].shift(lag)
        return lagged_data.dropna()

    def split_data(self):
        """Split data chronologically without shuffling."""
        split_index = int(len(self.data_base) * 0.8)
        X = self.data_base[self.features]
        y = self.data_base['position']
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    def train_with_gridsearch(self, model, param_grid):
        """Use GridSearchCV with time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=tscv)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_

    def is_imbalanced(self, threshold=0.2):
        """Check if the dataset is imbalanced based on a threshold."""
        value_counts = self.y_train.value_counts(normalize=True)
        imbalance_ratio = min(value_counts) / max(value_counts)
        return imbalance_ratio < threshold

    def run(self):
        # Handling class imbalance only if necessary
        if self.is_imbalanced():
            try:
                print("Imbalance detected. Applying SMOTE.")
                smote = SMOTE()
                X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
            except Exception as e:
                print(e)
                X_train_smote, y_train_smote = self.X_train, self.y_train
        else:
            print("No significant imbalance detected.")
            X_train_smote, y_train_smote = self.X_train, self.y_train

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(self.X_test)

        # Training models with GridSearchCV
        rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        best_rf_model = self.train_with_gridsearch(RandomForestClassifier(), rf_params)

        # Cross-validation scores
        cv_scores = cross_val_score(best_rf_model, X_train_scaled, y_train_smote, cv=5)
        print(f"RandomForest CV Scores: {cv_scores}")

        # AutoML with TPOT
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
        tpot.fit(X_train_scaled, y_train_smote)
        print(f"TPOT Score: {tpot.score(X_test_scaled, self.y_test)}")

        # Stacked Model
        estimators = [
            ('rf', best_rf_model),
            ('sgd', SGDClassifier(random_state=42)),
            ('svc', SVC(random_state=42)),
            ('mlp', MLPClassifier(random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=10))
        ]
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        stack.fit(X_train_scaled, y_train_smote)
        stacked_score = stack.score(X_test_scaled, self.y_test)
        print(f"Stacked Model Score: {stacked_score}")

        # Choosing the best model
        self.best_model = tpot if tpot.score(X_test_scaled, self.y_test) > stacked_score else stack
        predictions = self.best_model.predict(X_test_scaled)
        # Create a predictions DataFrame
        predictions_df = pd.DataFrame(index=self.X_test.index)
        predictions_df['pred_position'] = predictions
        predictions_df=predictions_df.join(self.data_base, how='inner')
        return predictions_df

