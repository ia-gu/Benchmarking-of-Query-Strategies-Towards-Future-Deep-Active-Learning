import torch
from torch.utils.data import Dataset
from .strategy import Strategy

class CoreSet(Strategy):
    
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based 
    approach using coreset selection. The embedding of each example is computed by the network’s 
    penultimate layer and the samples at each round are selected using a greedy furthest-first 
    traversal conditioned on all labeled examples.
    
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
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, cfg=None):
        super(CoreSet, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, cfg=cfg)
  
    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0]
                
        idxs = []
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])

            min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])

        return idxs
  
    def select(self, budget):
        
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        class NoLabelDataset(Dataset):
            
            def __init__(self, wrapped_dataset):
                self.wrapped_dataset = wrapped_dataset
                
            def __getitem__(self, index):
                features, _ = self.wrapped_dataset[index]
                return features
            
            def __len__(self):
                return len(self.wrapped_dataset)
        
        self.model.eval()
        embedding_unlabeled = self.get_embedding(self.unlabeled_dataset)
        embedding_labeled = self.get_embedding(NoLabelDataset(self.labeled_dataset))
        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)

        return chosen        