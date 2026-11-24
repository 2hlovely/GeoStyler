import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase,Datum_sf


@DATASET_REGISTRY.register()
class DomainNet_SF(DatasetBase):


    dataset_dir = "domainnet"
    domains = [
        "none","clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, cfg,train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets=[]
        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self._read_data([domain], split="all"))
        train=self._read_train_data(train_data)

        super().__init__(train_x=train, test=test_datasets)


    def _read_train_data(self,train_data):
        items=[]
        classnames=train_data["classnames"]
        n_cls=train_data["n_cls"]
        n_style=train_data["n_style"]
        for idx_cls in range(n_cls):
            for idx_style in range(n_style):
                item = Datum_sf(
                    cls=idx_cls,
                    style=idx_style,
                    label=idx_cls,
                    classname=classnames[idx_cls]
                )
                items.append(item)
        return items

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            splits = ["train", "test"] if split == "all" else [split]
            lines = []
            for split_name in splits:
                split_file = osp.join(
                    self.split_dir, dname + "_" + split_name + ".txt"
                )
                with open(split_file, "r") as f:
                    lines.extend((line, split_file) for line in f.readlines())
            seen = set()
            for line, split_file in lines:
                line = line.strip()
                relative_path, label = line.rsplit(maxsplit=1)
                if relative_path in seen:
                    continue
                seen.add(relative_path)
                path_parts = relative_path.split("/")
                if len(path_parts) < 3:
                    raise ValueError(
                        "Invalid DomainNet path '{}' in {}".format(
                            relative_path, split_file
                        )
                    )
                classname = path_parts[1]
                impath = osp.join(
                    self.image_dir, relative_path.replace("/", osp.sep)
                )
                if not osp.isfile(impath):
                    raise FileNotFoundError(
                        "DomainNet split entry does not exist: '{}' "
                        "(from {})".format(impath, split_file)
                    )
                label = int(label)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items
