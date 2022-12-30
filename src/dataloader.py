"""
處理專案的io物件
DataLoader: 處理資料io
"""
import os
import pandas as pd
import joblib
from pathlib import Path


class DataLoader():
    """
        This dataloader class provides a way to load the data and save data
        between working directory
        The following directories stores:
        train_data: all raw data (dp, custinfo , remit, ccba ,cdtx and so on)
        Preprocess: all the meta and precressing codes
        Model: the training code

        Args:
            file_path (str): project directory
            train_path (str): the directory path for train data
            input_path (str): the directory path for Preprocess
            model_path (str): the directory path for Model
        Returns:
            None
    """
    def __init__(self):
        """__init__ method"""
        file_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = file_path
        self.input_path = file_path / 'Preprocess'
        self.train_path = file_path / 'train_data'
        self.model_path = file_path / 'Model'

    def __repr__(self):
        """__repr__ method"""
        return "DataLoader(file_path = {})".format(self.file_path)

    def __str__(self):
        """__str__ method"""
        return "DataLoader(file_path = {})".format(self.file_path)

    def _load(self, file_path, data_type='csv'):
        """
        The load method 
        """
        if data_type == 'joblib':
            data = joblib.load(file_path)
        elif data_type == 'csv':
            data = pd.read_csv(file_path)
        elif data_type == 'excel':
            data = pd.read_excel(file_path)
        return data

    def _save(self, cls, file_path, cls_type='csv'):
        """
        The save method
        """
        if cls_type == 'csv':
            cls.to_csv(file_path, index=None)
        else:
            joblib.dump(cls, file_path)

    def load_input(self, data_name, data_type='joblib'):
        """load Preprocess method"""
        file_path = self.input_path / data_name
        return self._load(file_path, data_type)

    def load_train_data(self, data_name, data_type='csv'):
        """load train_data method"""
        file_path = self.train_path / data_name
        return self._load(file_path, data_type)

    def load_model(self, data_name, data_type='joblib'):
        """load Model method"""
        file_path = self.model_path / data_name
        return self._load(file_path, data_type)

    def save_input(self, cls, data_name, cls_type='joblib'):
        """save Preprocess method"""
        file_path = self.input_path / data_name
        self._save(cls, file_path, cls_type)

    def save_train_data(self, cls, data_name, cls_type='joblib'):
        """save train_data method"""
        file_path = self.train_path / data_name
        self._save(cls, file_path, cls_type)

    def save_model(self, cls, data_name, cls_type='joblib'):
        """save Model method"""
        file_path = self.model_path / data_name
        self._save(cls, file_path, cls_type)
