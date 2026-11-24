import os

import torch
import torch.nn as nn
import clip
from torch.nn import functional as F

exist = lambda target_path: os.path.exists(target_path)


class GeoStyler(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.classnames = classnames
        self.device = device
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model
        self.weight_save_path = os.path.join(cfg.TRAINER.GEOSTYLER.WEIGHT_DIR_PATH,
                                             cfg.TRAINER.GEOSTYLER.CHECK_POINT_NAME)

        self.prompt_style_template = "a X style of a"
        self.tokenized_prompt_style_template = clip.tokenize(self.prompt_style_template).to(self.device)
        with torch.no_grad():
            self.prompt_style_template_embedding = self.clip_model.token_embedding(
                self.tokenized_prompt_style_template)


        self.prompt_content_template = classnames

        self.tokenized_prompt_content_template = clip.tokenize(self.prompt_content_template).to(device)
        with torch.no_grad():
            self.prompt_content_output = self.clip_model.forward_text_ori(
                self.tokenized_prompt_content_template)
            self.prompt_content_output_norm = F.normalize(self.prompt_content_output, p=2, dim=1)


        self.init_template(self.classnames, "a X style of a CLS")
        self.tokenized_prompt_s_c = self.tokenized_template_base_text[0].to(self.device)
        self.prompt_s_c_embedding = self.template_base_text[0].to(self.device)
        self.train_flag = cfg.TRAINER.GEOSTYLER.TRAIN_STYLE

        with torch.no_grad():
            dummy_out = self.clip_model.forward_text(
                self.prompt_style_template_embedding,
                self.tokenized_prompt_style_template
            )
            text_dim = dummy_out.shape[1]
            rand_mat = torch.randn(
                text_dim,
                self.n_style,
                device=self.device,
                dtype=torch.float
            )
            q, _ = torch.linalg.qr(rand_mat)
            identity = torch.eye(
                self.n_style,
                device=self.device,
                dtype=torch.float
            )
            centering = identity - torch.full_like(
                identity,
                1.0 / self.n_style
            )
            simplex = centering * (
                self.n_style / (self.n_style - 1)
            ) ** 0.5
            self.etf_targets = simplex @ q.t()
        if not self.train_flag:
            self.style_embedding = nn.Parameter(
                torch.empty(self.n_style, 1, self.ctx_dim, dtype=torch.float))
            self.load_weight()
            self.style_embedding.requires_grad_(False)
            print(self.style_embedding.device)
        else:
            ctx_vectors = torch.empty(self.n_style, 1, self.ctx_dim, dtype=torch.float).to(self.device)
            ctx_vectors.requires_grad_(True)
            ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
            self.style_embedding = nn.Parameter(ctx_vectors)

    def save_style_embedding(self, weight_path):
        state_dict = self.state_dict()
        new_state_dict = {"style_embedding": state_dict["style_embedding"]}
        torch.save(new_state_dict, weight_path)
        print("save style embedding weight")

    def load_weight(self, device="cpu"):
        assert exist(self.weight_save_path), "prompt style weight path not exist!"
        state_dict = torch.load(self.weight_save_path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

        print(f"load weight from {self.weight_save_path}")

    def init_template(self, classnames, template_txt):
        self.template_base_text_list = []
        self.style_position = []
        self.tokenized_template_base_text = []
        self.template_base_text = []
        self.template_nums = 1
        template_txt_split = template_txt.split()
        style_position = template_txt_split.index("X")
        cls_position = template_txt_split.index("CLS")
        style_after_cls_flag = style_position > cls_position
        style_position_offset = [1 if "_" not in cls and not style_after_cls_flag else 3 for cls in classnames]

        template_base_text = [template_txt.replace('CLS', s) for s in classnames]
        tokenized_template_base_text = torch.cat([clip.tokenize(p) for p in template_base_text]).to(self.device)
        with torch.no_grad():
            template_base_text = self.clip_model.token_embedding(
                tokenized_template_base_text)
        self.tokenized_template_base_text.append(tokenized_template_base_text.to('cpu'))
        self.template_base_text.append(template_base_text.to('cpu'))

        self.template_base_text_list += template_base_text

        style_position = template_txt_split.index("X")
        self.style_position.append([style_position + i for i in style_position_offset])

    def get_stylized_embedding(self, template_idx, class_idx, style_idx):
        assert style_idx < len(self.style_embedding), "Style id is outside the length of the style list!"
        assert class_idx < len(self.classnames), "Class id is outside the length of the class list!"
        assert template_idx < self.template_nums, "Template id is outside the length of the template list!"
        token_template_base = self.tokenized_template_base_text[template_idx]
        template_base = self.template_base_text[template_idx]
        token_base = token_template_base[class_idx:class_idx + 1, :].clone()
        base = template_base[class_idx:class_idx + 1, :, :].clone()
        style_init_embedding = self.style_embedding[style_idx:style_idx + 1, :, :]
        style_position = self.style_position[template_idx][class_idx]
        base[:, style_position:style_position + 1, :] = style_init_embedding

        return base, token_base

    def forward(self, style_idx):
        current_style = self.style_embedding[style_idx:style_idx + 1, :, :]
        prefix = self.prompt_style_template_embedding[:, :2, :]
        suffix = self.prompt_style_template_embedding[:, 3:, :]
        current_prompt = torch.cat(
            [prefix, current_style, suffix],
            dim=1
        )
        current_prompt_output = self.clip_model.forward_text(
            current_prompt,
            self.tokenized_prompt_style_template
        )
        current_prompt_output_norm = F.normalize(
            current_prompt_output,
            p=2,
            dim=1
        )
        etf_difference = (
            current_prompt_output_norm.squeeze(0)
            - self.etf_targets[style_idx]
        )
        etf_loss = etf_difference.pow(2).sum()

        sc_prompts_list = []
        for cls_idx in range(self.n_cls):
            style_position_idx = self.style_position[0][cls_idx]
            sc_prefix = self.prompt_s_c_embedding[cls_idx:cls_idx + 1, :style_position_idx, :]
            sc_suffix = self.prompt_s_c_embedding[cls_idx:cls_idx + 1, style_position_idx + 1:, :]
            sc_prompts_list.append(
                torch.cat(
                    [
                        sc_prefix,
                        current_style,
                        sc_suffix,
                    ],
                    dim=1,
                )
            )
        sc_prompts = torch.cat(sc_prompts_list, dim=0)
        sc_prompts_output = self.clip_model.forward_text(
            sc_prompts, self.tokenized_prompt_s_c
        )
        sc_prompts_output_norm = F.normalize(sc_prompts_output, p=2, dim=1)
        zimm = F.cosine_similarity(
            sc_prompts_output_norm.unsqueeze(1), self.prompt_content_output_norm.unsqueeze(0), dim=2
        )
        exp_zimm = torch.exp(zimm)
        per_zimm = exp_zimm / exp_zimm.sum(dim=1, keepdim=True)

        content_consistency_loss_dj = -torch.log(per_zimm.diag()).mean()
        total_loss = etf_loss + content_consistency_loss_dj
        return etf_loss, content_consistency_loss_dj, total_loss
