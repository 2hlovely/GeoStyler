import datetime
import os.path as osp
import time
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from dassl.utils import (
    AverageMeter,
    MetricMeter,
    load_checkpoint,
    mkdir_if_missing,
    resume_from_checkpoint,
    save_checkpoint,
    tolist_if_not
)


class TrainerBase:
    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before initialization")
        if self.__dict__.get("_optims") is None:
            raise AttributeError("Cannot assign optimizer before initialization")
        if self.__dict__.get("_scheds") is None:
            raise AttributeError("Cannot assign scheduler before initialization")
        if name in self._models:
            raise ValueError("Duplicate model name: {}".format(name))
        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        registered = list(self._models.keys())
        if names is None:
            return registered
        names = tolist_if_not(names)
        for name in names:
            if name not in registered:
                raise ValueError("Unknown model name: {}".format(name))
        return names

    def save_model(self, epoch, directory, val_result=None, model_name=""):
        for name in self.get_model_names():
            optimizer = self._optims[name]
            scheduler = self._scheds[name]
            save_checkpoint(
                {
                    "state_dict": self._models[name].state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "val_result": val_result
                },
                osp.join(directory, name),
                model_name=model_name
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        if any(not osp.exists(osp.join(directory, name)) for name in names):
            print("No checkpoint found, train from scratch")
            return 0
        print(f"Found checkpoint at {directory}")
        start_epoch = 0
        for name in names:
            start_epoch = resume_from_checkpoint(
                osp.join(directory, name),
                self._models[name],
                self._optims[name],
                self._scheds[name]
            )
        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("No model directory supplied")
            return
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")
            checkpoint = load_checkpoint(model_path)
            self._models[name].load_state_dict(checkpoint["state_dict"])
            print(f"Loaded {model_path} at epoch {checkpoint['epoch']}")

    def set_model_mode(self, mode="train", names=None):
        for name in self.get_model_names(names):
            if mode == "train":
                self._models[name].train()
            elif mode in ("test", "eval"):
                self._models[name].eval()
            else:
                raise ValueError("Unknown model mode: {}".format(mode))

    def update_lr(self, names=None):
        for name in self.get_model_names(names):
            scheduler = self._scheds[name]
            if scheduler is not None:
                scheduler.step()

    def init_writer(self, log_dir):
        if self._writer is None:
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def model_backward_and_update(self, loss, names=None):
        names = self.get_model_names(names)
        for name in names:
            optimizer = self._optims[name]
            if optimizer is not None:
                optimizer.zero_grad()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN")
        loss.backward()
        for name in names:
            optimizer = self._optims[name]
            if optimizer is not None:
                optimizer.step()


class TrainerX(TrainerBase):
    def check_cfg(self, cfg):
        return None

    def before_train(self):
        directory = self.cfg.RESUME or self.cfg.OUTPUT_DIR
        self.start_epoch = self.resume_model_if_exist(directory)
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)
        self.time_start = time.time()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                batches_left = self.num_batches - self.batch_idx - 1
                batches_left += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(
                    seconds=int(batch_time.avg * batches_left)
                ))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))
            iteration = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, iteration)
            self.write_scalar("train/lr", self.get_current_lr(), iteration)
            end = time.time()

    def parse_batch_test(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def get_current_lr(self, names=None):
        name = self.get_model_names(names)[0]
        return self._optims[name].param_groups[0]["lr"]
