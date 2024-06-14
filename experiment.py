import os
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
from time import sleep

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
        optimizers = self.optimizers()

        real_img, _, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img.float(), labels=labels)
        train_loss = self.model.loss_function(
            *results,  # al_img.shape[0]/ self.num_train_imgs,
            batch_idx=batch_idx,
            labels=labels,
            losses=self.params["losses"],
            optimizers = optimizers,
            training = True,
            backward = self.manual_backward
        )



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
            batch_idx=batch_idx,
            labels=labels,
            losses=self.params["losses"],
            training = False
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    def on_validation_epoch_end(self) -> None:
        # self.sample_images()
        # self.save_dna_latent_space()

        self.save_pml_latent_space()
        self.calc_rf_and_align_pml_from_csvs()

        #self.update_V()
        pass

    def update_V(self):
        val_loader = self.trainer.datamodule.val_dataloader()
        for i, batch in enumerate(val_loader):
            images, _fps, labels = batch
            X, _, _ = self.model.encode(images.to(self.curr_device).float())
            if self.current_epoch >=1:
                print('Updating V....')
                self.model.update_V(X, labels)
                path = os.path.join(self.logger.log_dir, "Latent", f"v_tree.png")
                self.model.v_tree.render(path, w=183, units="mm")
            break

    def calc_rf_and_align(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()
        train_gt = self.trainer.datamodule.train_dataset.phylogeny
        val_gt = self.trainer.datamodule.val_dataset.phylogeny
        for loader, prefix, gt in zip([train_loader, val_loader], ['train', 'val'], [train_gt, val_gt]):
            for i, batch in enumerate(loader):
                real_img, _, labels = batch

                results = self.forward(real_img.float().cuda(), labels=labels)

                mu = results[2].mean(dim=0)
                labels = [x[0] for x in labels]

                #_X = mu - mu.mean(dim=1).unsqueeze(1)
                #cov_q = torch.matmul(_X.transpose(2, 1), _X)

                nrf, align = self.model.calc_RF_and_align(
                    mu=mu.T, labels=labels, gt=val_gt
                )


            self.log_dict({f"{prefix}_RF": nrf, f"{prefix}_align": align}, sync_dist=True)

    def calc_rf_and_align_pml(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()
        train_gt = self.trainer.datamodule.train_dataset.phylogeny
        val_gt = self.trainer.datamodule.val_dataset.phylogeny
        for loader, prefix, gt in zip([train_loader, val_loader], ['train', 'val'], [train_gt, val_gt]):
            n_chars = self.model.n_characters
            genus_sums = defaultdict(lambda: np.zeros(n_chars))
            genus_counts = defaultdict(lambda: 0)

            for i, batch in enumerate(loader):
                real_img, _, labels = batch
                labels = [x.split(' ')[0] for x in labels]
                results = self.forward(real_img.float().cuda(), labels=labels)
                mu = results[2].cpu().detach().numpy()

                for k, label in enumerate(labels):
                    genus_sums[label] += mu[k].squeeze()
                    genus_counts[label] += 1

                if i >= 5:
                    break

            mus = np.empty([len(genus_sums.keys()),n_chars])
            labels = []
            for i, label in enumerate(genus_sums.keys()):
                labels += [label]
                mus[i] = genus_sums[label] / genus_counts[label]


            nrf, align = self.model.calc_RF_and_align(
                mu=mus, labels=labels, gt=gt
            )

            self.log_dict({f"{prefix}_RF": nrf, f"{prefix}_align": align}, sync_dist=True)


    def calc_rf_and_align_pml_from_csvs(self):

        train_path = os.path.join(self.logger.log_dir, "Latent", f"train_mu.csv")
        train = pd.read_csv(train_path, index_col=0)
        val_path = os.path.join(self.logger.log_dir, "Latent", f"val_mu.csv")
        val = pd.read_csv(val_path, index_col=0)

        train = train.groupby(train.index.map(lambda x: x.split(' ')[0])).mean()
        val = val.groupby(val.index.map(lambda x: x.split(' ')[0])).mean()

        train_gt = self.trainer.datamodule.train_dataset.phylogeny
        val_gt = self.trainer.datamodule.val_dataset.phylogeny
        for df, _set, gt in zip([train, val], ['train', 'val'], [train_gt, val_gt]):
            nrf, align = self.model.calc_RF_and_align(
                mu=df.values, labels=df.index, gt=gt
            )

            self.log_dict({f"{_set}_RF": nrf, f"{_set}_align": align}, sync_dist=True)



    def save_dna_latent_space(self):
        print("Saving Latent Space...")

        train_loader = self.trainer.datamodule.train_dataloader()
        val_loader = self.trainer.datamodule.val_dataloader()
        test_loader = self.trainer.datamodule.test_dataloader()

        for loader, name in (
            (train_loader, "train"),
            (val_loader, "val"),
            (test_loader, "test"),
        ):
            inp_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_inp.npy")
            recons_path = os.path.join(
                self.logger.log_dir, "Latent", f"{name}_recons.npy"
            )
            mu_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_mu.npy")
            log_var_path = os.path.join(
                self.logger.log_dir, "Latent", f"{name}_log_var.npy"
            )
            fps_path = os.path.join(
                self.logger.log_dir,
                "Latent",
                f"{name}_fps.npy",
            )
            for i, batch in enumerate(loader):
                images, _fps, _ = batch
                recons, inp, mu, log_var = self.model(
                    images.to(self.curr_device).float()
                )
                n_classes = mu.shape[-1]
                mu = mu.reshape(-1, self.model.n_characters, n_classes)
                # A = mu - mu.mean(axis=1).unsqueeze(1)
                _X = (
                    mu
                    - mu.mean(dim=1).unsqueeze(1)
                    # + torch.exp(0.5 * log_var).unsqueeze(1)
                )
                cov_q = (
                    torch.matmul(_X.transpose(2, 1), _X)
                    + torch.eye(n_classes).cuda() * 1e-2
                )

                break
            np.save(inp_path, inp.detach().cpu().numpy())
            np.save(recons_path, recons.detach().cpu().numpy())
            np.save(mu_path, mu.detach().cpu().numpy())
            np.save(log_var_path, cov_q.detach().cpu().numpy())
            np.save(fps_path, _fps)

            plt.figure()
            sns.heatmap(cov_q[0].cpu().detach().numpy())
            plt.savefig(
                os.path.join(self.logger.log_dir, "Latent", f"{name}_cov_q.png")
            )
            plt.close()



    def save_pml_latent_space(self):
        print("Saving Latent Space...")
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            val_loader = self.trainer.datamodule.val_dataloader()
            test_loader = self.trainer.datamodule.test_dataloader()

            for loader, name in (
                (train_loader, "train"),
                (val_loader, "val"),
                #(test_loader, "test"),
            ):

                mu_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_mu.csv")
                examples_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_examples.csv")
                n_chars = self.model.n_characters
                species_sums = defaultdict(lambda: np.zeros(n_chars))
                avg_accuracy_sum = 0
                avg_accuracy_count = 0
                species_counts = defaultdict(lambda: 0)
                species_examples = defaultdict(lambda: [])
                sleep(0.1)
                for i, batch in enumerate(loader):
                    with torch.no_grad():
                        images, _fps, labels = batch
                        recons, inp, mu, log_var = self.model(
                            images.to(self.curr_device).float()
                        )
                        mu = mu.cpu().detach().numpy()
                        for k, label in enumerate(labels):
                            species_sums[label] += mu[k].squeeze()
                            species_counts[label] += 1
                            avg_accuracy_sum += (images.squeeze().argmax(dim=2) == recons.argmax(dim=2).cpu()).float().mean(dim=1).sum()
                            avg_accuracy_count += images.shape[0]
                            if species_counts[label] < 200:
                                species_examples[label] += [mu[k]]

                        if i == 0:
                            bs = recons.shape[0]
                            for j in range(min(10,bs)):
                                out = np.hstack([inp[j].cpu().detach().numpy(),recons[j].cpu().detach().numpy()])
                                plt.figure()
                                sns.heatmap(out)
                                plt.savefig(
                                os.path.join(self.logger.log_dir, "Latent", f"inp_recons_{name}_{j}.png")
                                )
                                plt.close()

                            samples = torch.tensor(mu)
                            unique_labels = list(np.unique(labels))
                            int_labels = torch.tensor([unique_labels.index(x) for x in labels])
                            int_labels = int_labels.view(int_labels.size(0), 1).expand(-1, samples.size(1))

                            unique_int_labels, labels_count = int_labels.unique(dim=0, return_counts=True)

                            avg_per_label = torch.zeros_like(unique_int_labels, dtype=torch.float).scatter_add_(0, int_labels, samples)
                            avg_per_label = avg_per_label / labels_count.float().unsqueeze(1)

                            X = avg_per_label
                            n_characters = X.shape[1]
                            n_classes = X.shape[0]

                            indices = [self.model.T_labels.index(x) for x in unique_labels]
                            T = torch.tensor(self.model.T)[indices][:, indices] + torch.eye(n_classes) * 1e-5
                            inv_T = T.inverse().float()

                            a_hat = torch.matmul(inv_T, X).sum(axis=0)/inv_T.sum()
                            _X = X - a_hat

                            cov_q = torch.matmul(_X, _X.T) / (n_classes - 1)
                            plt.figure()
                            sns.heatmap(cov_q.cpu().detach().numpy())
                            plt.savefig(
                                os.path.join(self.logger.log_dir, "Latent", f"{name}_cov_q2.png")
                            )
                            plt.close()
                        #if self.current_epoch >= 1:
                        #    wtf

                    sleep(0.1)

                means = []
                examples = []
                for species in species_sums.keys():
                    means += [[species, species_sums[species] / species_counts[species]]]
                    for j in range(len(species_examples[species])):
                        examples += [[species, species_examples[species][j]]]
                means = pd.DataFrame([x[1] for x in means], index = [x[0] for x in means])
                means.to_csv(mu_path)

                accuracy_path = os.path.join(self.logger.log_dir, "Latent", f"{name}_accuracy.csv")
                accuracy = avg_accuracy_sum / avg_accuracy_count
                accuracy = pd.DataFrame([accuracy.item()], columns=['top 1 accuracy'])
                accuracy.to_csv(accuracy_path)


                X = means.loc[self.model.T_labels].values
                n_characters = X.shape[1]
                n_classes = X.shape[0]

                indices = [self.model.T_labels.index(x) for x in means.index]
                T = self.model.T[indices][:, indices]
                inv_T = np.linalg.inv(T)

                a_hat = np.matmul(inv_T, X).sum(axis=0)/inv_T.sum()
                _X = X - a_hat
                cov_q = np.matmul(_X, _X.T) / (n_classes - 1)


                plt.figure()
                sns.heatmap(cov_q)
                plt.savefig(
                    os.path.join(self.logger.log_dir, "Latent", f"{name}_cov_q.png")
                )
                plt.close()

                #Plot T
                T = self.model.T
                plt.figure()
                sns.heatmap(T)
                plt.savefig(
                    os.path.join(self.logger.log_dir, "Latent", f"T.png")
                )
                plt.close()

                examples = pd.DataFrame([x[1].squeeze() for x in examples], index = [x[0] for x in examples])
                examples.to_csv(examples_path)
        except:
            print('Could not save latent space.')


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
            betas = self.params["betas"]
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
