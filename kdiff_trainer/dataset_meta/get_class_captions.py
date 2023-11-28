from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ClassCaptions:
    # for display purposes in demo grids
    demo_class_captions: List[str]
    # for cross-attn embedding
    embed_class_captions: List[str]
    uncond_class_ix: int
    # we deliberately model this mapping as a Tuple rather than a Callable, to make it easier to pickle (it's used in a closure that the dataloader copies to spawned subprocesses)
    dataset_label_to_canonical_label: Optional[Tuple[int, ...]]

def get_class_captions(
   classes_to_captions: str,
) -> ClassCaptions:
    if classes_to_captions == 'oxford-flowers':
        from kdiff_trainer.dataset_meta.oxford_flowers import flower_classes, nelorth_to_fatima
        labels_excl_uncond: List[str] = flower_classes
        dataset_label_to_canonical_label: Tuple[int, ...] = nelorth_to_fatima
    elif classes_to_captions == 'imagenet-1k':
        from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
        labels_excl_uncond: List[str] = class_labels
        dataset_label_to_canonical_label: Optional[Tuple[int, ...]] = None
    elif classes_to_captions == 'imagenet-100':
        from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
        labels_excl_uncond: List[str] = class_labels[:100]
        dataset_label_to_canonical_label: Optional[Tuple[int, ...]] = None
    elif classes_to_captions == 'imagenet-geese':
        from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
        # labels_excl_uncond: List[str] = [class_labels[99]]
        # okay maybe I should've used a callback instead of a sequence
        # dataset_label_to_canonical_label: Tuple[Optional[int], ...] = [[None]*99, 0]
        labels_excl_uncond: List[str] = class_labels[:100]
        dataset_label_to_canonical_label: Optional[Tuple[int, ...]] = None
    elif classes_to_captions == 'afhq':
        labels_excl_uncond: List[str] = ['cat', 'dog', 'wild']
        dataset_label_to_canonical_label: Optional[Tuple[int, ...]] = None
    else:
        raise ValueError(f"Never heard of classes_to_captions '{classes_to_captions}'")
    demo_class_captions: List[str] = [*labels_excl_uncond, '<UNCOND>']
    embed_class_captions: List[str] = [*labels_excl_uncond, '']
    uncond_class_ix: int = len(demo_class_captions)-1
    return ClassCaptions(
        demo_class_captions=demo_class_captions,
        embed_class_captions=embed_class_captions,
        uncond_class_ix=uncond_class_ix,
        dataset_label_to_canonical_label=dataset_label_to_canonical_label,
    )