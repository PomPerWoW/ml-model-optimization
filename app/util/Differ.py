import numpy as np
from abc import ABC, abstractmethod

class Differ(ABC):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    
    @abstractmethod
    def find_difference(self):
        pass

class ArrayDiffer(Differ):
    def __init__(self, r1_raw, r2_raw):
        if not isinstance(r1_raw, np.ndarray) or not isinstance(r2_raw, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        
        # Call the parent constructor
        super().__init__(self._parse_raw_array(r1_raw), self._parse_raw_array(r2_raw))

    def _parse_raw_array(self, raw_array):
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

        for obj1 in self.v1:
            closest_match = self._find_closest_match(obj1, self.v2)
            if closest_match:
                conf_diff = abs(obj1['confidence'] - closest_match['confidence'])
                box_diff = np.abs(np.array(obj1['box']) - np.array(closest_match['box']))
                
                differences.append([obj1['class'], f"{conf_diff:.4f}", list(box_diff)])
        
        return differences
    
class MatrixDiffer(Differ):
    def __init__(self, matrix1, matrix2, threshold=1e-2):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.threshold = threshold

    def find_difference(self):
        matrix1_np = np.array(self.matrix1)
        matrix2_np = np.array(self.matrix2)
        
        result_diff = matrix1_np - matrix2_np
        
        result_diff[np.abs(result_diff) < self.threshold] = 0
        
        return result_diff
