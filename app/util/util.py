import numpy as np
from collections import defaultdict
import itertools

class DotDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

def parse_raw_array(raw_array):
    parsed_objects = []
    for row in raw_array:
        class_name = row[0]
        confidence = float(row[1])
        box = list(map(float, row[2:6]))
        parsed_objects.append({
            'class': class_name,
            'confidence': confidence,
            'box': box
        })
    return parsed_objects

class Differ:
    def __init__(self, r1_raw, r2_raw):
        if not isinstance(r1_raw, np.ndarray) or not isinstance(r2_raw, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        
        self.r1 = parse_raw_array(r1_raw)
        self.r2 = parse_raw_array(r2_raw)

    def _find_closest_match(self, obj, obj_list):
        min_distance = float('inf')
        closest_match = None
        
        obj_conf = obj['confidence']
        obj_box = np.array(obj['box'])

        for o in obj_list:
            if o['class'] == obj['class']:
                o_conf = o['confidence']
                o_box = np.array(o['box'])

                conf_dist = (obj_conf - o_conf) ** 2
                box_dist = np.sum((obj_box - o_box) ** 2)
                total_dist = np.sqrt(conf_dist + box_dist)

                if total_dist < min_distance:
                    min_distance = total_dist
                    closest_match = o
        
        return closest_match
    
    def find_difference(self):
        differences = []

        for obj1 in self.r1:
            closest_match = self._find_closest_match(obj1, self.r2)
            if closest_match:
                conf_diff = abs(obj1['confidence'] - closest_match['confidence'])
                box_diff = np.abs(np.array(obj1['box']) - np.array(closest_match['box']))
                
                differences.append([obj1['class'], conf_diff, list(box_diff)])
        
        return differences