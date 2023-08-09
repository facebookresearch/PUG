"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import itertools
import json
import copy
import re
import csv
import os

def process_json(file, start_index, folder_name):
    # Opening JSON file
    f = open(file)
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Iterating through the json
    # list
    list_keys = data.keys()

    dict_all = {}
    all_factors = {}
    for key2, value2 in data.items():
        if type(value2) is list:
            dict_all[key2] = []
            for k in range(len(value2)):
                new_factor_value = {key2: value2[k]}
                dict_all[key2].append(new_factor_value)
    # Closing file
    f.close()

    # Generate all combinations of factors from th config file
    all_combination = [m for m in itertools.product(*dict_all.values())]
    
    return all_combination

if __name__ == '__main__':
    process_json("configs/configs_test.json")