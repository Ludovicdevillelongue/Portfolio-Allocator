import os
import pickle

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


class ModelData:
    def __init__(
            self,
            data_frequency:str,
            data_base: pd.DataFrame,
            look_back: int,
            strategy:str,
            training_proportion: float = 0.7,
            validation_proportion: float = 0.15,
            testing_proportion: float = 0.15
    ):
        self.data_frequency=data_frequency
        self.data_base = data_base
        self.look_back = look_back
        self.strategy=strategy
        self.training_proportion = training_proportion
        self.validation_proportion = validation_proportion
        self.testing_proportion = testing_proportion

    def create_data_array(self):
        data_raw = self.data_base.values  # convert to numpy array
        data = []
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - self.look_back):
            data.append(data_raw[index: index + self.look_back])
        data=np.array(data)
        return data


    def init_X(self):
        return self.data_array[:, :-1, :-1]


    def init_Y(self):
        return self.data_array[:, -1, -1].reshape(-1, 1)


    def set_X(self, start, end, data_type):
        if data_type =='training':
            X=self.scaler_X.fit_transform(self.X[start:end,:,:].reshape(-1, self.X[start:end,:,:].shape[-1])).\
                reshape(self.X[start:end,:,:].shape)
        else:
            X = self.scaler_X.transform(self.X[start:end,:,:].reshape(-1, self.X[start:end,:,:].shape[-1])).\
                reshape(self.X[start:end,:,:].shape)
        return X

    def set_Y(self, start, end, data_type):
        if data_type =='training':
            Y = self.scaler_Y.fit_transform(self.Y[start:end].reshape(-1, self.Y[start:end].shape[-1]))\
                .reshape(self.Y[start:end].shape)
        else:
            Y = self.scaler_Y.transform(self.Y[start:end].reshape(-1, self.Y[start:end].shape[-1])) \
                .reshape(self.Y[start:end].shape)
        return Y


    def init_scaler(self):
        """
        Type of scaler to rank-normalize asset characteristics into the interval (-1,1)
        The rank normalization is done on each day for the asset characteristic of each tickers
        Excess returns are not rank normalize, as in the paper
        Therefore this is not done per asset (all characteristics of an asset by day) but on all characetrstics for all asset
        """
        return MinMaxScaler(feature_range=(-1, 1))


    def init_train_data(self) -> int:
        """
        Instantiation of the training data
        """
        self.scaler_X = self.init_scaler()
        self.scaler_Y=self.init_scaler()
        start = 0
        end = int(len(self.data_array) * self.training_proportion)

        self.X_train = self.set_X(start, end, 'training')
        self.Y_train=self.set_Y(start, end, 'training')
        return end

    def init_valid_data(self) -> int:
        """
        Instantiation of the validation data
        """
        start = self.init_train_data()
        end = start + int(len(self.data_array) * self.validation_proportion)
        self.X_valid = self.set_X(start, end, 'validation')
        self.Y_valid=self.set_Y(start, end, 'validation')
        return end

    def init_test_data(self):
        """
        Instantiation of the testing data
        """
        start = self.init_valid_data()
        end = start + int(len(self.data_array) * self.testing_proportion)
        self.X_test= self.set_X(start, end, 'testing')
        self.Y_test=self.set_Y(start, end, 'testing')

    def compute(self):
        """
        Main method of the class, allowing to perform the steps and saving the arguments because the computation is long
        """
        assert (0 <= self.training_proportion <= 1)
        assert (0 <= self.validation_proportion <= 1)
        assert (0 <= self.testing_proportion <= 1)
        assert (self.training_proportion + self.validation_proportion + self.testing_proportion == 1)
        self.data_array=self.create_data_array()
        self.X = self.init_X()
        self.Y=self.init_Y()
        self.init_train_data()
        self.init_valid_data()
        self.init_test_data()
        self.save_args()
        return self.data_model


    def save_args(self):
        """
        As the main method of the class is long in debug mode, saving the data to reload after
        """
        self.data_model = {'X_train': self.X_train,'X_scaler':self.scaler_X, 'Y_train': self.Y_train,
                'Y_train_scaler':self.scaler_Y,'X_valid': self.X_valid, 'Y_valid' : self.Y_valid,
                'X_test': self.X_test, 'Y_test' : self.Y_test,
                'data_frequency':self.data_frequency["period"], 'initial_data':self.data_base,
                'look_back':self.look_back, 'strategy':self.strategy}




class DLData():
    def __init__(self, data_frequency, data_base, look_back, strategy):
        self.data_frequency=data_frequency
        self.data_base=data_base
        self.look_back=look_back
        self.strategy=strategy
        self.data_model=None

    def is_pickle_exists(self, file_path):
        return os.path.exists(file_path)

    def create_train_test(self):
        modeldata=ModelData(self.data_frequency, self.data_base, self.look_back, self.strategy)
        self.data_model=modeldata.compute()

class LSTM(DLData):
    def __init__(self, data_frequency, data_base, look_back, strategy):
        super().__init__(data_frequency, data_base, look_back, strategy)

    def fit(self):
        self.create_train_test()
        # From SVM, instantiate SVC classifier model instance
        svm_model = svm.SVC()

        # Fit the model to the data using the training data
        svm_model = svm_model.fit(self.data_model['X_train'], self.data_model['Y_train'])

        # Use the testing data to make the model predictions
        svm_pred = svm_model.predict(self.data_model['X_test'])

        # Review the model's predicted values
        return svm_pred




