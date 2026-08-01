import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["features"] = torch.stack(
        [item["features"] for item in dataset_items]
    )
    result_batch["labels"] = torch.tensor(
        [item["labels"] for item in dataset_items], dtype=torch.long
    )
    result_batch["utterance_id"] = [item["utterance_id"] for item in dataset_items]

    return result_batch
