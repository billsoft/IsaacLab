import numpy as np
d = np.load('D:/code/IsaacLab/projects/stereo_voxel/output/voxel/frame_000000_semantic.npz')
g = d['data']
CLASS_NAMES = {0:"free",1:"car",2:"bicycle",3:"motorcycle",4:"truck",5:"other-vehicle",
    6:"person",7:"bicyclist",8:"motorcyclist",9:"road",10:"parking",11:"sidewalk",
    12:"other-ground",13:"building",14:"fence",15:"vegetation",16:"trunk",17:"general-object"}
unique, counts = np.unique(g, return_counts=True)
print("=== Voxel class distribution ===")
for u, c in zip(unique, counts):
    name = CLASS_NAMES.get(int(u), f"unknown-{u}")
    print(f'  [{int(u):2d}] {name:<20s}: {int(c):6d}')
total_occ = int(np.sum(g > 0))
print(f'\nTotal occupied: {total_occ} / {g.size} ({total_occ/g.size*100:.1f}%)')
