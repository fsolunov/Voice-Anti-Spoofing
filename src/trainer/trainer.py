import torch
from tqdm.auto import tqdm

from src.metrics.eer import compute_eer
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Weight averages by sample count; the template's default n=1 gives
        # the last short batch the same weight as a full batch.
        batch_size = batch["labels"].shape[0]
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)

        for met in metric_funcs:
            metrics.update(met.name, met(**batch), n=batch_size)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """Evaluate one whole split and compute EER exactly once.

        EER is nonlinear: averaging per-batch EERs is mathematically wrong.
        This override retains the template loop but pools every score and label
        before calling the exact homework implementation.
        """

        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)
                all_scores.append(batch["scores"].detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        eer_fraction, _ = compute_eer(scores[labels == 1], scores[labels == 0])
        logs = self.evaluation_metrics.result()
        logs["EER"] = 100.0 * eer_fraction
        logs["Selection"] = logs["EER"] + 0.01 * logs.get("loss", 0.0)

        self.writer.set_step(epoch * self.epoch_len, part)
        self._log_scalars(self.evaluation_metrics)
        self.writer.add_scalar("EER", logs["EER"])
        self._log_batch(batch_idx, batch, part)
        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
