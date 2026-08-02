from pathlib import Path

import soundfile
import torch

from src.datasets.base_dataset import BaseDataset


class ASVSpoof2019LADataset(BaseDataset):
    PROTOCOL_NAMES = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    def __init__(
        self,
        data_root,
        split,
        sample_rate=16000,
        protocol_path=None,
        *args,
        **kwargs,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.data_root = self._resolve_la_root(Path(data_root))
        self.protocol_path = self._resolve_protocol(protocol_path)
        index = self._read_protocol()
        super().__init__(index=index, *args, **kwargs)

    @staticmethod
    def _resolve_la_root(path: Path):
        candidates = (path, path / "LA", path / "LA" / "LA")
        for candidate in candidates:
            if (candidate / "ASVspoof2019_LA_train").is_dir():
                return candidate

    def _resolve_protocol(self, protocol_path):
        if protocol_path is not None:
            path = Path(protocol_path)
        else:
            path = (
                self.data_root
                / "ASVspoof2019_LA_cm_protocols"
                / self.PROTOCOL_NAMES[self.split]
            )
        return path

    def _read_protocol(self):
        audio_dir = self.data_root / f"ASVspoof2019_LA_{self.split}" / "flac"
        index = []
        seen_ids = set()
        with self.protocol_path.open(encoding="utf-8") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.split()
                speaker_id, utterance_id, _, attack_id, label = fields
                seen_ids.add(utterance_id)
                index.append(
                    {
                        "path": str(audio_dir / f"{utterance_id}.flac"),
                        "label": int(label == "bonafide"),
                        "utterance_id": utterance_id,
                        "speaker_id": speaker_id,
                        "attack_id": attack_id,
                    }
                )
        return index

    @property
    def utterance_ids(self):
        return [entry["utterance_id"] for entry in self._index]

    def __getitem__(self, index):
        entry = self._index[index]
        audio, sample_rate = soundfile.read(
            entry["path"], dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(audio).mean(dim=1, keepdim=True).transpose(0, 1)
        item = {
            "features": waveform,
            "labels": entry["label"],
            "utterance_id": entry["utterance_id"],
        }
        return self.preprocess_data(item)
