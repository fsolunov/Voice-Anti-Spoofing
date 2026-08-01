import csv
import math

import torch
from tqdm.auto import tqdm

from src.metrics.eer import compute_eer
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """Checkpoint inference that writes one grader-compatible CSV per split."""

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.evaluation_dataloaders = dict(dataloaders)
        self.save_path = save_path
        self.metrics = metrics
        if not skip_model_load:
            self._from_pretrained(config.inferencer.from_pretrained)

    def run_inference(self) -> dict:
        logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs[part] = self._inference_part(part, dataloader)
        return logs

    def _inference_part(self, part, dataloader) -> dict:
        self.is_train = False
        self.model.eval()
        metric_tracker = None
        if self.metrics is not None:
            metric_tracker = MetricTracker(
                *[metric.name for metric in self.metrics["inference"]], writer=None
            )

        score_by_id = {}
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)
                batch.update(self.model(**batch))
                batch_size = batch["labels"].shape[0]
                if metric_tracker is not None:
                    for metric in self.metrics["inference"]:
                        metric_tracker.update(
                            metric.name, metric(**batch), n=batch_size
                        )

                scores = batch["scores"].detach().cpu()
                labels = batch["labels"].detach().cpu()
                for utterance_id, score in zip(batch["utterance_id"], scores.tolist()):
                    score_by_id[utterance_id] = score
                all_scores.append(scores)
                all_labels.append(labels)

        protocol_order = dataloader.dataset.utterance_ids

        output_path = self.save_path / self.cfg_trainer.submission_name
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, lineterminator="\n")
            for utterance_id in protocol_order:
                writer.writerow(
                    [utterance_id, format(score_by_id[utterance_id], ".10g")]
                )

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        eer, _ = compute_eer(scores[labels == 1], scores[labels == 0])
        result = metric_tracker.result() if metric_tracker is not None else {}
        result.update({"EER": 100.0 * eer, "submission": str(output_path)})
        return result
