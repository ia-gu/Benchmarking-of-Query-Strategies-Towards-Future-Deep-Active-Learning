import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):

    """
    Implementation of Random Sampling Strategy. This strategy is often used as a baseline, 
    where we pick a set of unlabeled points randomly.
    
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
        super(RandomSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, cfg=cfg)
        
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

        rand_idx = np.random.permutation(len(self.unlabeled_dataset))[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx