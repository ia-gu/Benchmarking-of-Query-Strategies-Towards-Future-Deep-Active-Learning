import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):

    """
    Implements KMeans Sampling selection strategy, the last layer embeddings are calculated for all the unlabeled data points. 
    Then the KMeans clustering algorithm is run over these embeddings with the number of clusters equal to the budget. 
    Then the distance is calculated for all the points from their respective centers. From each cluster, the point closest to 
    the center is selected to be labeled for the next iteration. Since the number of centers are equal to the budget, selecting 
    one point from each cluster satisfies the total number of data points to be selected in one iteration.
    
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
    cfg: DictConfig
        Specify additional parameters
        
        - **batch_size**: Batch size to be used inside strategy class (int, optional)
    """

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, cfg=None):
        super(KMeansSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, cfg=cfg)

    def select(self, budget):
        embeddings = self.get_embedding(self.unlabeled_dataset)
        embeddings = embeddings.to('cpu').detach().numpy().copy()
        
        # clustering
        cluster_learner = KMeans(n_clusters=budget)
        cluster_learner.fit(embeddings)
        
        # get cluster center and the nearest points
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)

        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(budget)])
        return q_idxs
