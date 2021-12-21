import multiprocessing as mp
import os
import pickle

import numpy as np
import PIL.Image as Image

from model.ss import get_ss_proposals
import argparse







parser = argparse.ArgumentParser(description='Parameters for generating imagenet proposals vis ss')
parser.add_argument('--root', type=str)
parser.add_argument('--dest', type=str)
parser.add_argument('--proc', type=int)

args = parser.parse_args()




imagenet_root = args.root
dest = args.dest
os.makedirs(os.path.dirname(dest), exist_ok=True)


processes_num = args.proc
class_names = sorted(os.listdir(os.path.join(imagenet_root)))
classes_num = len(class_names)
classes_per_process = classes_num // processes_num + 1

def process_one_class(q, process_id, classes_per_process, class_names, source_path):
    print("Process id: {} started".format(process_id))
    for i in range(process_id*classes_per_process, process_id*classes_per_process + classes_per_process):
        if i >= len(class_names):
            break
        class_name = class_names[i]
        filenames = sorted(os.listdir(os.path.join(source_path, class_name)))
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            img_path = os.path.join(source_path, class_name, filename)
            img = np.array(Image.open(img_path).convert('RGB'))

            proposal = get_ss_proposals(img, scale=300, sigma=0.9, min_size=100)
            q.put({base_filename:proposal})
        print("Process ", process_id, "processed class:", class_name)

q = mp.Queue()
processes = [mp.Process(target=process_one_class,
                        args=(q, process_id, classes_per_process, class_names, imagenet_root))
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
