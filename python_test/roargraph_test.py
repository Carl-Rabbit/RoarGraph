from roargraph import RoarGraph
import numpy as np
import time

dim = 128
n_sample_queries = 100
n_vectors = 10000
k_knn = 100

sample_queries = np.random.rand(n_sample_queries, dim).astype(np.float32)
vectors = np.random.rand(n_vectors, dim).astype(np.float32)

rg = RoarGraph(100, 10, 35, 500, k_knn, 64)

start = time.perf_counter()
rg.build(
    vectors,
    sample_queries,
)
end = time.perf_counter()
print(f'>>>> {end - start:6f}s')

recalls = []
times = []
topk_times = []
for i in range(100):
    query = np.random.rand(dim).astype(np.float32)
    start = time.perf_counter()
    res = rg.search(query, 10)
    end = time.perf_counter()
    times.append(end - start)

    start = time.perf_counter()
    # real topk using IP
    real_topk = np.argsort(-np.dot(vectors, query))[:10]
    end = time.perf_counter()
    topk_times.append(end - start)
    
    recall = len(set(real_topk) & set(res)) / 10
    recalls.append(recall)

print(np.mean(recalls))
print(np.mean(times))
print(np.mean(topk_times))