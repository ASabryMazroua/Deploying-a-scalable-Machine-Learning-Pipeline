# -*- coding: utf-8 -*-
"""
Created on Sat May 15 02:44:14 2021

@author: Ahmed
"""

import os
import logging

logging.basicConfig(
    filename='./logs/simple_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test():
    '''
    We can use this to write any Pytests to be used with Github Actions
    '''
    pass



if __name__ == "__main__":
    test()