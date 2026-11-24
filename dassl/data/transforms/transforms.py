from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST
}


def build_transform(cfg, is_train=False):
    transforms = [
        Resize(
            cfg.INPUT.SIZE,
            interpolation=INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        ),
        ToTensor()
    ]
    if "normalize" in cfg.INPUT.TRANSFORMS:
        transforms.append(Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD
        ))
    return Compose(transforms)
