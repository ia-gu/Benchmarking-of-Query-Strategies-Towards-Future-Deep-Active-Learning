from src.active_learning_strategies import BADGE, EntropySampling, RandomSampling, LeastConfidenceSampling, \
                                        MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
                                        BALDDropout, BatchBALDDropout, ClusterMarginSampling

def get_strategy(train_dataset, labeled_dataset, net, num_classes, cfg):
    if cfg.strategy == 'badge':
        strategy = BADGE(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'entropy_sampling':
        strategy = EntropySampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'margin_sampling':
        strategy = MarginSampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'least_confidence':
        strategy = LeastConfidenceSampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'coreset':
        strategy = CoreSet(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'random_sampling':
        strategy = RandomSampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'bald_dropout':
        strategy = BALDDropout(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'adversarial_bim':
        strategy = AdversarialBIM(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'kmeans_sampling':
        strategy = KMeansSampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'adversarial_deepfool':
        strategy = AdversarialDeepFool(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'batch_bald':
        strategy = BatchBALDDropout(train_dataset, labeled_dataset, net, num_classes, cfg)
    elif cfg.strategy == 'cluster_margin':
        strategy = ClusterMarginSampling(train_dataset, labeled_dataset, net, num_classes, cfg)
    else:
        raise IOError('Enter Valid Strategy!')
    return strategy