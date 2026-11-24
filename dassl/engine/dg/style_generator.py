import os

import clip
import torch
from torch import nn


class BaseStyleGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.classnames = classnames
        self.device = device
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.clip_model = clip_model
        base_texts = ["a random style of a " + name for name in classnames]
        self.style_position = [1 for _ in classnames]
        self.tokenized_base_text = torch.cat([
            clip.tokenize(text) for text in base_texts
        ]).to(device)
        self.base_embedding = clip_model.token_embedding(
            self.tokenized_base_text
        ).to("cpu")
        self.style_embedding = None

    def get_stylized_embedding(self, base_embedding, style_position, style_id):
        if style_id >= len(self.style_embedding):
            raise IndexError("Style index is out of range")
        embedding = base_embedding.clone()
        embedding[0, style_position:style_position + 1, :] = (
            self.style_embedding[style_id].clone()
        )
        return embedding

    def train_data(self):
        return {
            "classnames": self.classnames,
            "base_embedding": self.base_embedding,
            "tokenized_base_text": self.tokenized_base_text.to("cpu"),
            "style_generator": self,
            "n_cls": len(self.classnames),
            "n_style": self.n_style,
            "style_position": self.style_position
        }


class GeoStylerGenerator(BaseStyleGenerator):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)
        self.weight_save_path = os.path.join(
            cfg.TRAINER.GEOSTYLER.WEIGHT_DIR_PATH,
            cfg.TRAINER.GEOSTYLER.CHECK_POINT_NAME
        )
        self.load_weight()

    def reinit_style(self):
        return None

    def load_weight(self, device="cpu"):
        if not os.path.exists(self.weight_save_path):
            raise FileNotFoundError(self.weight_save_path)
        state_dict = torch.load(self.weight_save_path, map_location=device)
        self.style_embedding = state_dict["style_embedding"]
        self.style_embedding.requires_grad_(False)
        print(f"Loaded styles from {self.weight_save_path}")
