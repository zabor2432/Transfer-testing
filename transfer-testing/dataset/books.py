import os
from typing import Callable, Optional, Tuple, List, Dict, Any
from torchvision.datasets import DatasetFolder

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


class BooksDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(
        self, dir: str, augmented_book=False
    ) -> Tuple[List[str], Dict[str, int]]:
        # TODO: see how we can use this to work on our data structure
        if augmented_book:
            classes = sorted(
                entry.name for entry in os.scandir(dir) if entry.is_dir()
            )
        else:
            classes = sorted(
                entry.name for entry in os.scandir(dir) if entry.is_dir()
            )

        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {dir}."
            )

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
