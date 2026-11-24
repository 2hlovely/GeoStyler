import datetime
import time
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from dassl.data import DataManager_sf
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param, load_pretrained_weights
from dassl.engine.dg.style_generator import BaseStyleGenerator, GeoStylerGenerator
from dassl.modeling import build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import AngularPenaltySMLoss
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
        t = self.backbone.forward_text(x, t_x)
        t = t / t.norm(dim=-1, keepdim=True)
        if self.head is not None:
            t = self.head(t)
        return t

    def forward_img(self, x):
        t_img = self.backbone.forward_image(x)
        t_img = t_img / t_img.norm(dim=-1, keepdim=True)
        if self.head is not None:
            t_img = self.head(t_img)
        return t_img


class clip_net_arcface(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface'):
        super(clip_net_arcface, self).__init__()
        self.embedlayers = clip_net(cfg, model_cfg, device)
        in_features = self.embedlayers.fdim
        self.adms_loss = AngularPenaltySMLoss(in_features, num_classes, loss_type=loss_type, s=cfg.ARCFACE_S,
                                              m=cfg.ARCFACE_M)

    def forward_text(self, x, t_x, stylize_base_text_embedding, labels, norm=False):
        text_encoder_output = self.embedlayers.forward_text(x, t_x)
        if norm:
            text_encoder_output = text_encoder_output / text_encoder_output.norm(dim=-1, keepdim=True)
        y_pred = self.adms_loss.fc(text_encoder_output)
        y_loss = self.adms_loss(text_encoder_output, labels)

        y_domain = self.predictor(text_encoder_output,
                                  stylize_base_text_embedding) if stylize_base_text_embedding is not None else None
        return y_pred, y_loss, y_domain, text_encoder_output

    def forward_img(self, x, norm=False):
        t_img = self.embedlayers.forward_img(x)
        if norm:
            t_img = t_img / t_img.norm(dim=-1, keepdim=True)
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


        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        self.build_model()
        self.build_train_data()
        self.build_data_loader()

        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = ([], 0)

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
        else:
            raise Exception(f"{self.cfg.DATASET.NAME} dataset is not supported!")

        self.model = clip_net_arcface(cfg, cfg.MODEL, self.num_classes, self.device)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)


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
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    @torch.no_grad()
    def build_style_text_features(self):
        base_embedding = self.style_generator.base_embedding.to(self.device)
        tokenized_base_text = self.style_generator.tokenized_base_text.to(self.device)
        style_embedding = self.style_generator.style_embedding.to(self.device)
        style_position = self.style_generator.style_position
        n_style = style_embedding.size(0)
        n_cls = len(self.lab2cname)
        text_features = []
        for style_idx in range(n_style):
            prompts = []
            for cls_idx in range(n_cls):
                base = base_embedding[cls_idx:cls_idx + 1].clone()
                position = style_position[cls_idx]
                base[:, position:position + 1, :] = style_embedding[
                    style_idx:style_idx + 1
                ]
                prompts.append(base)
            prompts = torch.cat(prompts, dim=0)
            features = self.model.embedlayers.backbone.forward_text(
                prompts,
                tokenized_base_text
            )
            features = features / features.norm(dim=-1, keepdim=True)
            text_features.append(features)
        self.style_text_features = torch.stack(text_features, dim=0)

    @torch.no_grad()
    def test(self, split=None):

        self.set_model_mode("eval")
        self.evaluator.reset()
        result = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        split = "test"
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
        last_epoch = self.epoch + 1 == self.max_epoch
        checkpoint_freq = self.cfg.TRAIN.CHECKPOINT_FREQ
        if checkpoint_freq > 0 and (self.epoch + 1) % checkpoint_freq == 0:
            self.save_model(self.epoch, self.output_dir)
        if last_epoch:
            self.save_model(
                self.epoch,
                self.output_dir,
                model_name="model-best.pth.tar"
            )

    def after_train(self):
        print("Finish training")
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()

    def train(self):
        self.before_train()
        self.style_generator.reinit_style()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def model_inference(self, input):
        return self.model_inference_clip(input)

    def model_inference_clip(self, input):
        with torch.no_grad():
            input = input.to(self.device)

            img_feat = self.model.embedlayers.backbone.forward_image(input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            if not hasattr(self, "style_text_features"):
                self.build_style_text_features()
            S, C, D = self.style_text_features.shape
            logits_all = torch.einsum("nd,scd->snc", img_feat, self.style_text_features)
            probs_all = logits_all.softmax(dim=-1)
            ent = -(probs_all * probs_all.log()).sum(dim=-1)
            K = min(self.cfg.DYNA_STYLER.TOPK_STYLES, S)
            entropy_ns = ent.permute(1, 0)
            _, topk_idx = entropy_ns.topk(K, dim=-1, largest=False)
            probs_ns_c = probs_all.permute(1, 0, 2)
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)
            selected_probs = torch.gather(probs_ns_c, 1, gather_idx)
            selected_entropy = torch.gather(entropy_ns, 1, topk_idx)
            tau = self.cfg.DYNA_STYLER.TEMPERATURE
            weights = torch.softmax(
                -selected_entropy / tau,
                dim=1
            ).unsqueeze(-1)
            style_probs = (selected_probs * weights).sum(dim=1)
            base_logits = self.model.adms_loss.fc(img_feat)
            base_probs = base_logits.softmax(dim=-1)
            alpha = self.cfg.DYNA_STYLER.ALPHA
            final_probs = alpha * base_probs + (1 - alpha) * style_probs

        return final_probs

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
