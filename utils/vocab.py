"""
vocabulary
"""
import typing


class Vocabulary(object):

    def __init__(self, index_base: int=0):

        self.index_base = index_base

        self.values = []
        self.value2idx = dict()

    def add(self, value):
        if value in self.value2idx:
            return self.value2idx[value]

        else:
            idx = self.index_base + len(self.values)

            self.values.append(value)
            self.value2idx[value] = idx

            return idx

    def __getitem__(self, key: int):

        return self.values[key - self.index_base]

    def index(self, value):
        return self.value2idx[value]

