import json
import numpy as np
import argparse
from sklearn.cluster import KMeans


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--embedding_path', type=str, default="save/openhermes/embeddings/openhermes_emb_llama3.npy")
    argparse.add_argument('--instruction_path', type=str, default="datasets/openhermes/openhermes_random_10w.json")
    argparse.add_argument('--save_path', type=str, default="datasets/openhermes/openhermes_kmeans_llama3_100.json")
    argparse.add_argument('--new_path', type=str, default="datasets/openhermes/openhermes_random_10w_clustered.json")
    argparse.add_argument('--n_clusters', type=int, default=10)
    args = argparse.parse_args()

    print(args)
    EMBEDDING_PATH = args.embedding_path
    INSTRUCTION_PATH = args.instruction_path
    SAVE_PATH = args.save_path


    embeddings = []
    embeddings.extend(np.load(f'{EMBEDDING_PATH}'))
    embeddings = embeddings[:100]
    print(len(embeddings))
    print("K-MEANS")

    # KMeans clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(embeddings)

    # kmeans.cluster_centers_
    def find_nearest(embedding, embeddings):
        distances = ((embeddings - embedding) ** 2).sum(axis=1)
        return distances.argmin()
    
    cluster_center_indices = [find_nearest(center, embeddings) for center in kmeans.cluster_centers_]

    print(cluster_center_indices)

    data = []
    with open(f"{INSTRUCTION_PATH}", "r") as f:
        # data = json.load(f)
        for line in f.readlines():
            data.append(json.loads(line))

    labels = list(kmeans.labels_)
    print(len(data))

    import copy
    new_data = []
    for i in range(len(data)):
        tmp = copy.deepcopy(data[i])
        tmp['id'] = i
        tmp['cluster'] = int(labels[i])
        tmp['cluster_center'] = int(cluster_center_indices[labels[i]])
        new_data.append(tmp)

    print(len(new_data))
    # tmp = json.dumps(data, ensure_ascii=False, indent=4)
    with open(args.new_path, 'w') as f:
        for line in new_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
        # json.dump(new_data, f, ensure_ascii=False, indent=4)

    kmeans_sample = [new_data[i] for i in cluster_center_indices]

    kmeans_sample = json.dumps(kmeans_sample, indent=4)
    with open(f'{SAVE_PATH}', 'w') as f:
        f.write(kmeans_sample)
