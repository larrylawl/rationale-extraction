import multiprocessing as mp
import torch

done = mp.Event()

def extractor_worker(done_queue):
    done_queue.put(torch.Tensor(10,10).numpy())
    done_queue.put(None)
    done.wait()

producers = []
done_queue = mp.Queue()
for i in range(0, 1):
    process = mp.Process(target=extractor_worker,
                         args=(done_queue,))
    process.start()
    producers.append(process)

result_arrays = []
nb_ended_workers = 0
while nb_ended_workers != 1:
    worker_result = done_queue.get()
    if worker_result is None:
        nb_ended_workers += 1
    else:
        result_arrays.append(worker_result)
done.set()
print(result_arrays)