import multiprocessing as mp
import os
import pickle

import numpy as np
import PIL.Image as Image

from model.ss import get_ss_proposals
import argparse
import tqdm


parser = argparse.ArgumentParser(description='Parameters for generating coco proposals vis ss')
parser.add_argument('--root', type=str)
parser.add_argument('--dest', type=str)
parser.add_argument('--proc', type=int)

args = parser.parse_args()




coco_root = args.root
dest = args.dest
if os.path.dirname(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)


processes_num = args.proc
filenames = sorted(os.listdir(os.path.join(coco_root)))
print("Start processing {} imgs".format(len(filenames)))
files_num = len(filenames)
files_per_process = files_num // processes_num + 1

def process_one_class(q, process_id, files_per_process, filenames, source_path):
    print("Process id: {} started".format(process_id))
    for i in tqdm.tqdm(range(process_id*files_per_process, process_id*files_per_process + files_per_process), disable=(process_id !=0 )):
        if i >= len(filenames):
            break
        filename = filenames[i]
        img_path = os.path.join(source_path, filename)
        img = np.array(Image.open(img_path).convert('RGB'))

        proposal = get_ss_proposals(img, scale=300, sigma=0.9, min_size=100)
        q.put({filename:proposal})
        if i % 100 == 0:
            print("processing {}".format(filename))

q = mp.Queue()
processes = [mp.Process(target=process_one_class,
                        args=(q, process_id, files_per_process, filenames, coco_root))
                        for process_id in range(processes_num)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()
    
res = {}
while not q.empty():
    res.update(q.get())
with open(dest, 'wb') as f:
    pickle.dump(res, f)
