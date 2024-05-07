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

# [['car' '0.9299132823944092' '558' '206' '808' '359']
#  ['car' '0.9205407500267029' '286' '210' '458' '352']
#  ['car' '0.9116201996803284' '465' '217' '596' '339']
#  ['person' '0.8726341724395752' '159' '143' '301' '403']
#  ['truck' '0.8677411079406738' '103' '90' '255' '316']
#  ['truck' '0.7794504761695862' '722' '170' '871' '346']
#  ['truck' '0.7472285628318787' '0' '154' '94' '354']
#  ['bicycle' '0.651627779006958' '210' '321' '266' '443']
#  ['car' '0.5186470150947571' '78' '212' '113' '300']
#  ['car' '0.36083880066871643' '420' '226' '474' '319']
#  ['car' '0.3002343475818634' '420' '227' '464' '278']]

# [['car' '0.9305105805397034' '558' '206' '808' '359']
#  ['car' '0.9209166169166565' '286' '210' '458' '352']
#  ['car' '0.9119920134544373' '465' '217' '596' '339']
#  ['person' '0.8731658458709717' '159' '143' '301' '403']
#  ['truck' '0.8680006265640259' '103' '89' '255' '316']
#  ['truck' '0.7941577434539795' '722' '170' '871' '346']
#  ['truck' '0.7463480830192566' '0' '154' '94' '354']
#  ['bicycle' '0.6512946486473083' '210' '321' '266' '443']
#  ['car' '0.5251938104629517' '78' '212' '113' '300']
#  ['car' '0.36326032876968384' '420' '226' '474' '319']
#  ['car' '0.2982402443885803' '420' '227' '464' '278']]

class Differ:
    def __init__(self, r1, r2):
        if not isinstance(r1, np.ndarray) or not isinstance(r2, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")

        self.r1 = r1
        self.r2 = r2

    def find_differences(self):
        list1 = [tuple(row) for row in self.r1]
        list2 = [tuple(row) for row in self.r2]

        class_dict1 = defaultdict(list)
        class_dict2 = defaultdict(list)

        for entry in list1:
            class_dict1[entry[0]].append(entry)

        for entry in list2:
            class_dict2[entry[0]].append(entry)

        matched_pairs = []

        for classname, entries1 in class_dict1.items():
            if classname in class_dict2:
                entries2 = class_dict2[classname]

                for e1 in entries1:
                    best_match = min(entries2, key=lambda e2: abs(float(e1[1]) - float(e2[1])))
                    matched_pairs.append((e1, best_match))
                    entries2.remove(best_match)

        differences = []

        for pair in matched_pairs:
            e1, e2 = pair
            classname = e1[0]
            conf_diff = abs(float(e1[1]) - float(e2[1]))

            bbox_diffs = np.abs(np.array(e1[2:6]).astype(np.float32) - np.array(e2[2:6]).astype(np.float32))

            diff_dict = {
                "classname": classname,
                "conf_diff": conf_diff,
                "bbox_diffs": bbox_diffs.tolist(),
            }

            differences.append(diff_dict)

        return differences