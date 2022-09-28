from .score_streaming_strategy import ScoreStreamingStrategy

class LeastConfidenceSampling(ScoreStreamingStrategy):
    
    """
    Implements the Least Confidence Sampling Strategy a active learning strategy where 
    the algorithm selects the data points for which the model has the lowest confidence while 
    predicting its label.
    
    Suppose the model has `nclasses` output nodes denoted by :math:`\\overrightarrow{\\boldsymbol{z}}` 
    and each output node is denoted by :math:`z_j`. Thus, :math:`j \\in [1, nclasses]`. 
    Then for a output node :math:`z_i` from the model, the corresponding softmax would be 
    
    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} 
        
    Then the softmax can be used pick `budget` no. of elements for which the model has the lowest 
    confidence as follows, 
    
    .. math::
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    
    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    
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
        super(LeastConfidenceSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, cfg=cfg)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        U = -probs.max(1)[0] # Max prob. negated => Largest score corresponds to smallest max prob (least confident prediction)
        return U