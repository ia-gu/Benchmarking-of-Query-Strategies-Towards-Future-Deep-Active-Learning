from .strategy import Strategy
import numpy as np

import torch
from torch import nn
from scipy import stats

def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class BADGE(Strategy):
    '''
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient 
    Lower Bounds :footcite:`DBLP-Badge`. According to the paper, this strategy, Batch Active 
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate 
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to 
    incorporate both predictive uncertainty and sample diversity into every selected batch. 
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned 
    hyperparameters. Here at each round of selection, loss gradients are computed using the 
    hypothesised labels. Then to select the points to be labeled are selected by applying 
    k-means++ on these loss gradients.
    
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
    '''
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, cfg=None): 
        super(BADGE, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, cfg=cfg)        
        
    def select(self, budget):
        '''
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        '''	

        self.model.eval()
        gradEmbedding = self.get_grad_embedding(self.unlabeled_dataset, True, 'fc')
        chosen = init_centers(gradEmbedding.cpu().numpy(), budget, self.device)
        return chosen