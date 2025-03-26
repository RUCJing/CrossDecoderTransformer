# -*- coding: utf-8 -*-
import pickle
from abc import ABCMeta, abstractmethod


class FileOperator(metaclass=ABCMeta):
    def __init__(self, data, file_path):
        self.data = data
        self.file_path = file_path

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def save(self):
        pass


# pickle file operator
class PickleFileOperator(FileOperator):
    def __init__(self, data=None, file_path=''):
        super(PickleFileOperator, self).__init__(data, file_path)

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def read(self):
        with open(self.file_path, "rb") as f:
            content = pickle.load(f)
        return content



class ModelFileOperator(FileOperator):
    def __init__(self, data=None, file_path=''):
        super(ModelFileOperator, self).__init__(data, file_path)

    def save(self):
        pass

    def read(self):
        pass
