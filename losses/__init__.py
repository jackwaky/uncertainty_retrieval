from losses.bi_batch_based_classification_loss import BiBatchBasedClassificationLoss
from losses.batch_based_classification_loss import BatchBasedClassificationLoss
from losses.uncertainty_loss import AleatoricLoss, BatchBasedAleatoricLoss

def loss_factory(config):
    if config['metric_loss'] == BatchBasedClassificationLoss.code():
        return {
            'metric_loss': BatchBasedClassificationLoss(),
        }
    elif config['metric_loss'] == AleatoricLoss.code():
        return {
            'metric_loss': AleatoricLoss(),
        }
    elif config['metric_loss'] == BatchBasedAleatoricLoss.code():
        return {
            'metric_loss': BatchBasedAleatoricLoss(config['epoch'], config['gamma_scale']),
        }
    elif config['metric_loss'] == BiBatchBasedClassificationLoss.code():
        return {
            'metric_loss': BiBatchBasedClassificationLoss(),
        }
    else:
        raise ValueError("Expected metric loss function, but got {}".format(config['metric_loss']))