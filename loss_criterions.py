from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss


def get_loss_criterion(loss_name):
    match loss_name:
        case 'dice':
            return DiceLoss
        case 'dice_ce':
            return DiceCELoss
        case 'dice_focal':
            return DiceFocalLoss
