import json
import numpy as np
import copy

class ExtendedTxt(object):
    def __init__(self, arrays={}, meta={}):
        self.arrays = arrays
        self.meta = meta
    def save(self, extt_file):
        save_extt(extt_file, self.arrays, self.meta)
    def load(self, extt_file):
        self.arrays, self.meta = load_extt(extt_file)
    def clone(self):
        return ExtendedTxt(
            arrays={ a: np.copy(arr) for a, arr in self.arrays.items() },
            meta=copy.deepcopy(self.meta))
    def __getitem__(self, key):
        return self.arrays[key]

def save_extt(extt_file, arrays, meta={}):
    if type(arrays) is dict:
        with open(extt_file, 'w') as f:
            for k, v in arrays.items():
                f.write('%s:%s ' % (k, str(v.shape).replace(" ","")))
            f.write('\n')
            f.write(json.dumps(meta))
            f.write('\n')
            for k, v in arrays.items(): # order is guaranteed to match loop above
                np.savetxt(f, v)
            f.close()
    else:
        arrays.save(extt_file)
    return

def load_extt(extt_file):
    with open(extt_file) as f:
        tuples = [ item.split(":") \
            for item in f.readline().strip().split() ]
        array_rows = { t[0]: int(t[1].replace("(","").split(",")[0]) \
            for t in tuples }
        meta_str = f.readline().strip()
        meta = json.loads(meta_str)
        arrays = {}
        for array_name, n_rows in array_rows.items():
            arrays[array_name] = np.loadtxt(f, max_rows=n_rows)
    return ExtendedTxt(arrays=arrays, meta=meta)

