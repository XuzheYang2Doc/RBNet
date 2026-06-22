"""Rice leaf lesion segmentation dataset for MMSegmentation."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module(name='MyDataset')
@DATASETS.register_module()
class RiceLeafLesionDataset(BaseSegDataset):
    """Binary semantic segmentation dataset for rice blast lesions."""

    METAINFO = dict(
        classes=('background', 'lesion'),
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    def __init__(
        self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

