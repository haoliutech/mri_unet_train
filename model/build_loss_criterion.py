from loss.loss import *

def build_criterion(config=None, criterion_name=None):
    
    if criterion_name is None:
        kwargs = dict(config['loss_criterion'])
        criterion_name=kwargs.pop('loss_function', None)

    else:
        kwargs={}

    LossModule = {
        'TverskyLoss' : TverskyLoss,
        'TverskyFocalLoss' : TverskyFocalLoss,
        'FBetaLoss' : FBetaLoss,
        'DiceScore' : DiceScore,
        'BCELoss' : BCELoss, 
        'FocalLoss': FocalLoss,
        'DiceLoss' : DiceLoss, 
        'WeightedDiceLoss' : WeightedDiceLoss,
        'BCEDiceLoss' : BCEDiceLoss,
    }

    assert criterion_name in LossModule.keys(), f'criterion: {criterion_name} not valid'

    print(f'criterion selected: {criterion_name}')
    
    return LossModule[criterion_name](**kwargs)
