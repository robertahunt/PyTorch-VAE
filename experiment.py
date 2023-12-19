import os
import math
import torch
import numpy as np
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.automatic_optimization = False
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.save_hyperparameters(params)
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        real_img, _, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img.float(), labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
            batch_idx=batch_idx,
            labels=labels,
            phylo_weight=self.params["phylo_weight"],
            phylo_size=self.params["phylo_size"],
            phylo_mean=self.params["phylo_mean"],
            recons_weight=self.params["recons_weight"],
            triplet_weight=self.params["triplet_weight"],
        )

        self.manual_backward(train_loss["loss"])
        opt.step()

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx):
        real_img, _, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img.float(), labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # real_img.shape[0]/ self.num_val_imgs,
            batch_idx=batch_idx,
            labels=labels,
            phylo_weight=self.params["phylo_weight"],
            phylo_size=self.params["phylo_size"],
            phylo_mean=self.params["phylo_mean"],
            recons_weight=self.params["recons_weight"],
            triplet_weight=self.params["triplet_weight"],
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    def on_validation_epoch_end(self) -> None:
        # self.sample_images()
        # self.save_latent_space()
        self.save_dna_latent_space()

    def save_dna_latent_space(self):
        print("Saving Latent Space...")
        latent_dim = self.model.latent_dim

        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        for loader, name in (
            (train_loader, "train"),
            (val_loader, "val"),
            (test_loader, "test"),
        ):
            mu_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_mu.npy")
            log_var_path = os.path.join(
                self.logger.log_dir, "Latent", f"{name}_log_var.npy"
            )
            fps_path = os.path.join(
                self.logger.log_dir,
                "Latent",
                f"{name}_fps.npy",
            )
            mus = np.empty([0, 6, latent_dim])
            log_vars = np.empty([0, latent_dim, latent_dim])
            fps = np.empty([0, latent_dim])
            for i, batch in enumerate(loader):
                images, _fps, _ = batch
                mu, log_var = self.model.encode(images.to(self.curr_device).float())
                mus = np.concatenate([mus, mu.detach().cpu().numpy()])
                log_vars = np.concatenate([log_vars, log_var.detach().cpu().numpy()])
                fps = np.concatenate([fps, np.array(_fps).squeeze().transpose(1, 0)])
                break
            np.save(mu_path, mus)
            np.save(log_var_path, log_vars)
            np.save(fps_path, fps)

    def save_latent_space(self):
        print("Saving Latent Space...")
        latent_dim = self.model.latent_dim
        if self.trainer.datamodule.__class__.__name__ in [
            "BIOSCANVAEDataset",
            "BIOSCANTreeVAEDataset",
        ]:
            train_loader = self.trainer.datamodule.train_dataloader()
        else:
            train_loader = self.trainer.datamodule.train_dataloader_val_transformed()
        val_loader = self.trainer.datamodule.val_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        for loader, name in (
            (train_loader, "train"),
            (val_loader, "val"),
            (test_loader, "test"),
        ):
            mu_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_mu.npy")
            log_var_path = os.path.join(
                self.logger.log_dir, "Latent", f"{name}_log_var.npy"
            )
            fps_path = os.path.join(
                self.logger.log_dir,
                "Latent",
                f"{name}_fps.npy",
            )
            mus = np.empty([0, latent_dim])
            log_vars = np.empty([0, int((latent_dim**2 + latent_dim) / 2)])
            fps = np.empty([0, latent_dim])
            for i, batch in enumerate(loader):
                images, _fps, _ = batch
                mu, log_var = self.model.encode(images.to(self.curr_device).float())
                mus = np.concatenate([mus, mu.detach().cpu().numpy()])
                log_vars = np.concatenate([log_vars, log_var.detach().cpu().numpy()])
                fps = np.concatenate([fps, np.array(_fps).squeeze().transpose(1, 0)])
            np.save(mu_path, mus)
            np.save(log_var_path, log_vars)
            np.save(fps_path, fps)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _, test_label = next(
            iter(self.trainer.datamodule.test_dataloader())
        )
        test_input = test_input.to(self.curr_device)
        try:
            test_label = test_label.to(self.curr_device)
        except:
            pass

        #         test_input, test_label = batch
        recons = self.model.generate(test_input.float(), labels=test_label)

        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=12,
        )
        vutils.save_image(
            test_input.data,
            os.path.join(
                self.logger.log_dir,
                "Inputs",
                f"input_{self.logger.name}.png",
            ),
            normalize=True,
            nrow=12,
        )

        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            if len(samples.shape) == 3:
                samples = samples.unsqueeze(1)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir,
                    "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params["LR_2"] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.params["submodel"]).parameters(),
                    lr=self.params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params["scheduler_gamma"] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.params["scheduler_gamma"]
                )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params["scheduler_gamma_2"] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.params["scheduler_gamma_2"]
                        )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
