from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import numpy as np
import random

from .strategy import Strategy
from torch.utils.data import Subset


class ClusterMarginSampling(Strategy):

    """
    Implements Cluster Margin Sampling from the paper: Batch Active Learning at Scale
    Cluster Margin Sampling works even large batch size.

    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
    """

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        self.first = True
        self.features = None
        self.cluster_size = [0]*int((len(labeled_dataset)+len(unlabeled_dataset))/1.25)
        self.stream_buffer_size = len(unlabeled_dataset)
        super(ClusterMarginSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)

    def select(self, budget: int):
        self.model.eval()

        # HAC clustering only once
        if self.first:
            self.first = False
            self.features = self.get_embedding(self.unlabeled_dataset)
            self.features = self.features.to('cpu').detach().numpy().copy()
            self.clustered_features = AgglomerativeClustering(n_clusters=int(len(self.unlabeled_dataset)/1.25), affinity='cosine', linkage='average').fit(self.features)

        # Calculate size of each cluster
        for i in self.clustered_features.labels_:
            self.cluster_size[i] += 1
        self.cluster_size = np.argsort(self.cluster_size)

        # Gain margin
        evaluated_points = 0
        buffered_stream = Subset(self.unlabeled_dataset, list(range(evaluated_points, min(len(self.unlabeled_dataset), evaluated_points + self.stream_buffer_size))))
        probs = self.predict_prob(buffered_stream)
        probs_sorted, _ = probs.sort(descending=True)
        batch_scores = probs_sorted[:, 0] - probs_sorted[:, 1]
        batch_scores = [(x, i + evaluated_points) for i,x in enumerate(batch_scores)]
        clustered_samples = list()
        for i in range(len(self.cluster_size)):
            clustered_samples.append(list())

        # Conbine cluster and margin, then shuffle content
        for i in range(len(self.clustered_features.labels_)):
            clustered_samples[self.clustered_features.labels_[i]].append(batch_scores[i])
        for i in range(len(clustered_samples)):
            random.shuffle(clustered_samples[i])

        # Sort data based on margin metric
        batch_scores = sorted(batch_scores, key=lambda x: x[0])

        selected_cluster = []
        for i in range(len(batch_scores)):
            cluster_id = self.clustered_features.labels_[batch_scores[i][1]]
            if cluster_id in selected_cluster:
                pass
            else:
                selected_cluster.append(cluster_id)
        
        # Sort clusters based on cluster_size with minimum margin metric
        sorted_cluster_ids = []
        for i in range(len(self.cluster_size)):
            for j in range(len(self.cluster_size)):
                if (self.cluster_size[j] == i) and (j in selected_cluster):
                    sorted_cluster_ids.append(j)
                    break

        # sampling
        selected_ids = []
        while evaluated_points < budget:
            for i in sorted_cluster_ids:
                if evaluated_points >= budget:
                    break
                if len(clustered_samples[i]) == 0:
                    continue
                selected_ids.append(clustered_samples[i][0][1])
                clustered_samples[i].pop(0)
                evaluated_points += 1

        self.clustered_features.labels_ = np.delete(self.clustered_features.labels_, selected_ids)

        return selected_ids
