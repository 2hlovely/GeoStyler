import datetime
import time
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import clip
from collections import OrderedDict
from dassl.data import DataManager, DataManager_sf
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.engine.dg.style_generator import RandomStyleGenerator, BaseStyleGenerator, MixStyleGenerator, \
    RandomMixStyleGenerator, GeoStylerGenerator
from dassl.modeling import build_head, build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import AngularPenaltySMLoss, EntropyMaximization, InfoNCE
from dassl.evaluation import build_evaluator


class clip_net(nn.Module):
    def __init__(self, cfg, model_cfg, device, **kwargs):
        super().__init__()

        self.device = device
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
            **kwargs
        )
        self.fdim = self.backbone._out_features
        self.head = None
        

    def forward_text(self, x, t_x):
        t = self.backbone.forward_text(x, t_x)  # text embed without norm
        t = t / t.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t = self.head(t)  # text embed after head without norm
        return t

    def forward_img(self, x):  # for test
        t_img = self.backbone.forward_image(x)
        t_img = t_img / t_img.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t_img = self.head(t_img)  # img embed after head without norm
        return t_img


class clip_net_arcface(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface'):
        super(clip_net_arcface, self).__init__()
        self.embedlayers = clip_net(cfg, model_cfg, device)
        in_features = self.embedlayers.fdim  # embed dim
        self.adms_loss = AngularPenaltySMLoss(in_features, num_classes, loss_type=loss_type, s=cfg.ARCFACE_S,
                                              m=cfg.ARCFACE_M)

    def forward_text(self, x, t_x, stylize_base_text_embedding, labels, norm=False):
        text_encoder_output = self.embedlayers.forward_text(x, t_x)  # without normalize after head
        if norm:
            text_encoder_output = text_encoder_output / text_encoder_output.norm(dim=-1, keepdim=True)  # norm
        y_pred = self.adms_loss.fc(text_encoder_output)
        y_loss = self.adms_loss(text_encoder_output, labels)

        y_domain = self.predictor(text_encoder_output,
                                  stylize_base_text_embedding) if stylize_base_text_embedding is not None else None
        return y_pred, y_loss, y_domain, text_encoder_output

    def forward_img(self, x, norm=False):
        t_img = self.embedlayers.forward_img(x)  # without normalize after head
        if norm:
            t_img = t_img / t_img.norm(dim=-1, keepdim=True)  # norm
        y_class = self.adms_loss.fc(t_img)
        return y_class

    def predictor(self, feat, teat):
        feat_p = feat / feat.norm(dim=-1, keepdim=True)
        teat_p = teat / teat.norm(dim=-1, keepdim=True)
        scores = (100.0 * torch.matmul(feat_p, teat_p.detach().T))
        scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], 1)
        return scores


@TRAINER_REGISTRY.register()
class WOPA_clip(TrainerX):
    def __init__(self, cfg):
        # super().__init__(cfg)
        self.style_generator: BaseStyleGenerator = None
        self.num_classes = None
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        self.use_dyna_styler = hasattr(cfg, "DYNA_STYLER") and getattr(cfg.DYNA_STYLER, "ENABLE", False)

        self.build_model()
        self.build_train_data()
        self.build_data_loader()

        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = ([0, 0, 0, 0], 0)

        self.enm_loss = EntropyMaximization()

        self.infonce_loss = InfoNCE(negative_mode='paired')

    def build_model(self):
        cfg = self.cfg
        print("Building model")

        if self.cfg.DATASET.NAME == 'PACS_SF':
            self.num_classes = 7
        elif self.cfg.DATASET.NAME == 'OfficeHomeDG_SF':
            self.num_classes = 65
        elif self.cfg.DATASET.NAME == 'VLCS_SF':
            self.num_classes = 5
        elif self.cfg.DATASET.NAME == 'DomainNet_SF':
            self.num_classes = 345
        elif self.cfg.DATASET.NAME == 'TerraIncognita_SF':
            self.num_classes = 10
        elif self.cfg.DATASET.NAME == 'ImageNetR':
            self.num_classes = 200
        elif self.cfg.DATASET.NAME == 'ImageNetR_SF':
            self.num_classes = 200
        elif self.cfg.DATASET.NAME == 'ImageNetS_SF':
            self.num_classes = 1000
        else:
            raise Exception(f"{self.cfg.DATASET.NAME} dataset is not supported!")

        self.model = clip_net_arcface(cfg, cfg.MODEL, self.num_classes, self.device)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        
        # 诊断输出：确认模型在 GPU 上
        print(f"\n{'='*50}")
        print(f"Model Device Check:")
        print(f"  Target device: {self.device}")
        print(f"  Backbone device: {next(self.model.embedlayers.backbone.parameters()).device}")
        print(f"  CLIP visual device: {next(self.model.embedlayers.backbone.model.visual.parameters()).device}")
        print(f"  ArcFace fc device: {self.model.adms_loss.fc.weight.device}")
        print(f"{'='*50}\n")
        
        print(f"# adms_loss params: {count_num_param(self.model.adms_loss):,}")
        if self.model.embedlayers.head is not None:
            print(f"# head params: {count_num_param(self.model.embedlayers.head):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

    def forward_backward(self, batch_data):
        input_stylized_embedding, input_tokenized_base_text, target = self.parse_batch_train(batch_data)
        y_pred, y_loss, y_domain, _ = self.model.forward_text(input_stylized_embedding, input_tokenized_base_text,
                                                              None, target,
                                                              norm=True)
        loss = y_loss

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "y_loss": y_loss.item(),
            "acc": compute_accuracy(y_pred, target)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input_stylized_embedding = batch["stylized_embedding"]
        input_tokenized_base_text = batch["tokenized_base_text"]
        target = batch["label"]
        input_stylized_embedding = input_stylized_embedding.to(self.device)
        input_tokenized_base_text = input_tokenized_base_text.to(self.device)
        target = target.to(self.device)

        return input_stylized_embedding, input_tokenized_base_text, target

    def build_data_loader(self):
        train_data = self.style_generator.train_data()
        dm = DataManager_sf(self.cfg, train_data)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm

        if self.use_dyna_styler:
            classnames = [self.lab2cname[i] for i in range(len(self.lab2cname))]
            base_embedding = self.style_generator.base_embedding.to(self.device)  # [n_cls, L, dim]
            tokenized_base_text = self.style_generator.tokenized_base_text.to(self.device)  # [n_cls, L]
            style_embedding = self.style_generator.style_embedding.to(self.device)  # [n_style, 1, dim]
            style_position = self.style_generator.style_position  # list of length n_cls
            n_style = style_embedding.size(0)
            n_cls = len(classnames)
            text_feats_per_style = []
            with torch.no_grad():
                for s in range(n_style):
                    sc_prompts_list = []
                    for cls_idx in range(n_cls):
                        base_emb = base_embedding[cls_idx:cls_idx + 1].clone()
                        pos = style_position[cls_idx]
                        base_emb[:, pos:pos + 1, :] = style_embedding[s:s + 1]
                        sc_prompts_list.append(base_emb)
                    sc_prompts = torch.cat(sc_prompts_list, dim=0)
                    feats_s = self.model.embedlayers.backbone.forward_text(sc_prompts, tokenized_base_text)
                    feats_s = feats_s / feats_s.norm(dim=-1, keepdim=True)
                    text_feats_per_style.append(feats_s)  # [n_cls, dim]
            self.style_text_features = torch.stack(text_feats_per_style, dim=0)  # [n_style, n_cls, dim]

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        result = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()
        mean_acc = np.mean(result)
        is_best = False
        if self.best_result[-1] < mean_acc:
            self.best_result = (result, mean_acc)
            is_best = True
        return result, is_best


    def after_epoch(self):

        result, is_best = self.test()
        if is_best:
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=result,
                model_name="model-best.pth.tar"
            )
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

    def train(self):
        self.before_train()
        self.style_generator.reinit_style()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            if self.epoch % 2 == 0:
                self.after_epoch()
        self.after_train()

    def model_inference(self, input):
        if self.use_dyna_styler:
            return self.model_inference_clip(input)
        return self.model.forward_img(input, norm=True)

    def model_inference_clip(self, input):
        with torch.no_grad():
            input = input.to(self.device)
            
            img_feat = self.model.embedlayers.backbone.forward_image(input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [N, dim]
            
            # style_text_features: [S, C, dim]
            S, C, D = self.style_text_features.shape
            logits_all = torch.einsum("nd,scd->snc", img_feat, self.style_text_features)
            probs_all = logits_all.softmax(dim=-1)  # [S, N, C]
            
            ent = -(probs_all * probs_all.log()).sum(dim=-1)  # [S, N]
            
            use_gap = getattr(self.cfg.DYNA_STYLER, "USE_GAP", True)
            if use_gap:
                top2_vals, _ = probs_all.topk(2, dim=-1)  # [S, N, 2]
                gap = top2_vals[..., 0] - top2_vals[..., 1]  # [S, N]
                score = ent - gap  
            else:
                score = ent  
                gap = None
            
            selection_mode = getattr(self.cfg.DYNA_STYLER, "SELECTION_MODE", "ent_gap")
            K = min(self.cfg.DYNA_STYLER.TOPK_STYLES, S)
            
            if selection_mode == "confidence":
                confidence_ns = (-ent).permute(1, 0)  
                _, topk_idx = confidence_ns.topk(K, dim=-1, largest=True)  # [N, K]
                gap = -ent  
            else:
                score_ns = score.permute(1, 0)  # [N, S]
                _, topk_idx = score_ns.topk(K, dim=-1, largest=False) 
            
            if self.model.embedlayers.head is not None:
                img_feat_after_head = self.model.embedlayers.head(img_feat)
            else:
                img_feat_after_head = img_feat
            
            base_logits = self.model.adms_loss.fc(img_feat_after_head)  # [N, C]
            
            logits_sn_c = logits_all.permute(1, 0, 2)  # [N, S, C]
            
            use_weighted = getattr(self.cfg.DYNA_STYLER, "USE_WEIGHTED_ENSEMBLE", False)
            
            if use_weighted:
                if gap is None:
                    top2_vals, _ = probs_all.topk(2, dim=-1)
                    gap = top2_vals[..., 0] - top2_vals[..., 1]
                gap_ns = gap.permute(1, 0)  # [N, S]
                tau = getattr(self.cfg.DYNA_STYLER, "TEMPERATURE", 1.0)
                
                ensembled_style_logits = []
                for n in range(img_feat.size(0)):
                    idx_s = topk_idx[n]  # [K]
                    logits_nk = logits_sn_c[n, idx_s, :]  # [K, C]
                    gaps_nk = gap_ns[n, idx_s]  # [K]
                    
                    alpha = F.softmax(gaps_nk / tau, dim=0).unsqueeze(-1)  # [K, 1]
                    
                    weighted_logits = (logits_nk * alpha).sum(dim=0)  # [C]
                    ensembled_style_logits.append(weighted_logits)
                
                style_logits = torch.stack(ensembled_style_logits, dim=0)  # [N, C]
            else:
                ensembled_style_logits = []
                for n in range(img_feat.size(0)):
                    idx_s = topk_idx[n]  # [K]
                    logits_nk = logits_sn_c[n, idx_s, :]  # [K, C]
                    ensembled_style_logits.append(logits_nk.mean(dim=0))
                style_logits = torch.stack(ensembled_style_logits, dim=0)  # [N, C]
            
            alpha = getattr(self.cfg.DYNA_STYLER, "ALPHA", 0.5)
            logits_final = alpha * base_logits + (1 - alpha) * style_logits
            
        return logits_final

    def build_train_data(self):

        txts_dir_path = self.cfg.TXTS_PATH
        txt_path = os.path.join(txts_dir_path, self.cfg.DATASET.NAME + '.txt')

        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        classnames = list(class_dict.values())
        self.num_classes = len(classnames)
        assert self.cfg.STYLE_GENERATOR.NAME in globals()
        self.style_generator = globals()[self.cfg.STYLE_GENERATOR.NAME](self.cfg, classnames,
                                                                        self.model.embedlayers.backbone, self.device)
