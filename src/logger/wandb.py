from datetime import datetime

import numpy as np
import pandas as pd


class WandBWriter:
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): name of the entity inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        try:
            import wandb
        except ImportError as error:
            raise ImportError("Install project requirements to use WandB") from error

        mode = str(mode).lower()
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "writer.mode must be one of: online, offline, disabled; "
                f"got {mode!r}"
            )

        # An empty value is convenient in Hydra but W&B expects either a real
        # account/team name or Python None.
        entity = entity or None

        # Offline and disabled smoke tests must not contact the W&B API or ask
        # for credentials. Online runs need an explicit owner on accounts that
        # do not have a default entity configured.
        if mode == "online":
            if entity is None:
                raise ValueError(
                    "Online W&B logging requires an entity. Set "
                    "WANDB_ENTITY to your W&B username/team or override "
                    "writer.entity=<name>."
                )
            wandb.login()

        trainer_config = project_config.get("trainer", {})
        is_resuming = trainer_config.get("resume_from") is not None
        # A fresh run must not query W&B for resume status. In offline mode the
        # SDK cannot resume a cloud run, so resume is enabled only for an
        # explicit online checkpoint continuation.
        resume = "allow" if mode == "online" and is_resuming else None

        self.run_id = run_id
        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            resume=resume,
            id=self.run_id,
            mode=mode,
            save_code=kwargs.get("save_code", True),
        )
        self.wandb = wandb
        # These summaries make the best dev result visible on the run page.
        wandb.define_metric("loss_dev", summary="min")
        wandb.define_metric("EER_dev", summary="min")
        wandb.define_metric("Accuracy_dev", summary="max")

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the files section
        inside the run_name dir.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | ndarray | Image): image in the WandB-friendly
                format.
        """
        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)}, step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the WandB-friendly format.
            sample_rate (int): audio sample rate.
        """
        audio = audio.detach().cpu().numpy().T
        self.wandb.log(
            {
                self._object_name(audio_name): self.wandb.Audio(
                    audio, sample_rate=sample_rate
                )
            },
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)}, step=self.step
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.

        Args:
            hist_name (str): name of the histogram to use in the tracker.
            values_for_hist (Tensor): array of values to calculate
                histogram of.
            bins (int | str): the definition of bins for the histogram.
        """
        values_for_hist = values_for_hist.detach().cpu().numpy()
        np_hist = np.histogram(values_for_hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(values_for_hist, bins=512)

        hist = self.wandb.Histogram(np_histogram=np_hist)

        self.wandb.log({self._object_name(hist_name): hist}, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.

        Args:
            table_name (str): name of the table to use in the tracker.
            table (DataFrame): table content.
        """
        self.wandb.log(
            {self._object_name(table_name): self.wandb.Table(dataframe=table)},
            step=self.step,
        )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
