from torch.optim.lr_scheduler import StepLR


class EpochStepLR(StepLR):
    def __init__(
        self,
        optimizer,
        steps_per_epoch: int,
        decay_every_epochs: int = 10,
        gamma: float = 0.5,
        last_epoch: int = -1,
    ):
        super().__init__(
            optimizer=optimizer,
            step_size=steps_per_epoch * decay_every_epochs,
            gamma=gamma,
            last_epoch=last_epoch,
        )
