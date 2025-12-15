import torch

class TemporalWaggleCollator:
    """
        Collate function to load the dataset into data loaders
    """
    def __call__(self, batch):
        videos = [sample['video'] for sample in batch]
        targets = [sample['targets'] for sample in batch]
        labels = [sample['label'] for sample in batch]
        metadata = [sample['metadata'] for sample in batch]

        return {
            "video": torch.stack(videos),
            "targets": torch.stack(targets),
            "label": torch.tensor(labels),
            "metadata": metadata
        }