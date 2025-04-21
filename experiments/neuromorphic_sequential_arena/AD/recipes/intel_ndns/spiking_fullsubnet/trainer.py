import pandas as pd
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer import Trainer as BaseTrainer
import torch

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.sisnr_loss = SISNRLoss(return_neg=False)
        self.si_sdr = SISDR()
        self.north_star_metric = "si_sdr"
        self.all_samples_firing_rate = []
        self.all_samples_neurons_firing_rate = []
        self.all_samples_synops = []
    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        noisy_y, clean_y, _ = batch

        batch_size, *_ = noisy_y.shape

        enhanced_y, enhanced_mag, *_ = self.model(noisy_y)

        loss_freq_mae = freq_MAE(enhanced_y, clean_y)
        loss_mag_mae = mag_MAE(enhanced_y, clean_y)
        loss_sdr = self.sisnr_loss(enhanced_y, clean_y)
        loss_sdr_norm = 0.001 * (100 - loss_sdr)
        loss = loss_freq_mae + loss_mag_mae + loss_sdr_norm  # + loss_g_fake

        self.accelerator.backward(loss)
        self.optimizer.step()

        return {
            "loss": loss,
            "loss_freq_mae": loss_freq_mae,
            "loss_mag_mae": loss_mag_mae,
            "loss_sdr": loss_sdr,
            "loss_sdr_norm": loss_sdr_norm,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mix_y, ref_y, id = batch
        # print(f"id: {id}")
        est_y, enh_mag, fb_all_layer_outputs, sb_all_layer_outputs = self.model(mix_y)
        # torch.save(torch.stack(fb_all_layer_outputs).cpu(), self.exp_dir / f"{id[0].split('_')[-1]}_fb_all_layer_outputs_ref_y.ckpt")
        # torch.save(torch.stack(fb_all_layer_forgetgates).cpu(), self.exp_dir / f"{id[0].split('_')[-1]}_fb_all_layer_forgetgates_ref_y.ckpt")

        self.compute_firing_rate_per_sample = False
        self.compute_firing_rate_per_neuron = False
        self.compute_synops_per_sample = False
        if self.compute_firing_rate_per_sample:
            total_spikes = 0.
            total_neurons = 0.
            T = 3751
            for fb_layer_i in range(len(fb_all_layer_outputs)):
                if fb_layer_i + 1 < len(fb_all_layer_outputs):
                    # print(f"fb_all_layer_outputs: {fb_all_layer_outputs[fb_layer_i].size()}")
                    total_spikes = total_spikes + fb_all_layer_outputs[fb_layer_i].sum()
                    total_neurons = total_neurons + fb_all_layer_outputs[fb_layer_i].size(-2) * fb_all_layer_outputs[fb_layer_i].size(-1)
                    assert fb_all_layer_outputs[fb_layer_i].size(0) == T
            for sb_layer_i in range(len(sb_all_layer_outputs)):
                # torch.save(torch.stack(sb_all_layer_outputs[sb_layer_i]).cpu(), self.exp_dir / f"{id[0].split('_')[-1]}_sb_all_layer_{sb_layer_i}_outputs_ref_y.ckpt")
                # torch.save(torch.stack(sb_all_layer_forgetgates[sb_layer_i]).cpu(), self.exp_dir / f"{id[0].split('_')[-1]}_sb_all_layer_{sb_layer_i}_forgetgates_ref_y.ckpt")
                for sb_layer_i_j in range(len(sb_all_layer_outputs[sb_layer_i])):
                    if sb_layer_i_j + 1 < len(sb_all_layer_outputs[sb_layer_i]):
                        # print(f"sb_all_layer_outputs: {sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size()}")

                        total_spikes = total_spikes + sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].sum()
                        total_neurons = total_neurons + sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(-2) * sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(-1)
                        assert sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(0) == T
                # print(f"sb_all_layer_outputs: {torch.stack().size()}")
            sample_firing_rate = total_spikes / (total_neurons * T)
            # print(sample_firing_rate)
            # print(f"total_spikes: {total_spikes}")
            self.all_samples_firing_rate.append(sample_firing_rate.item())
        elif self.compute_firing_rate_per_neuron:
            total_spikes = 0.
            total_neurons = 0.
            T = 3751
            all_neurons_firing_rate = []
            for fb_layer_i in range(len(fb_all_layer_outputs)):
                if fb_layer_i + 1 < len(fb_all_layer_outputs):
                    # print(f"fb_all_layer_outputs: {fb_all_layer_outputs[fb_layer_i].size()}")
                    spike_patterns = fb_all_layer_outputs[fb_layer_i].view(T, -1)
                    # print(spike_patterns.mean(0).size())
                    all_neurons_firing_rate.append(spike_patterns.mean(0))
                    assert fb_all_layer_outputs[fb_layer_i].size(0) == T
            for sb_layer_i in range(len(sb_all_layer_outputs)):
                for sb_layer_i_j in range(len(sb_all_layer_outputs[sb_layer_i])):
                    if sb_layer_i_j + 1 < len(sb_all_layer_outputs[sb_layer_i]):
                        spike_patterns = sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].view(T, -1)
                        all_neurons_firing_rate.append(spike_patterns.mean(0))
                        # print(spike_patterns.mean(0).size())
                        assert sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(0) == T
            self.all_samples_neurons_firing_rate.append(torch.cat(all_neurons_firing_rate))
            # print(self.all_samples_neurons_firing_rate[0].size())
            # exit()

        elif self.compute_synops_per_sample:
            synops_per_sample = 0.
            T = 3751
            for fb_layer_i in range(len(fb_all_layer_outputs)):
                if fb_layer_i + 1 < len(fb_all_layer_outputs):
                    synops_per_sample = synops_per_sample + (fb_all_layer_outputs[fb_layer_i].sum() / T ) * (fb_all_layer_outputs[fb_layer_i].size(-1) + fb_all_layer_outputs[fb_layer_i+1].size(-1))
                    assert fb_all_layer_outputs[fb_layer_i].size(0) == T
            for sb_layer_i in range(len(sb_all_layer_outputs)):
                for sb_layer_i_j in range(len(sb_all_layer_outputs[sb_layer_i])):
                    if sb_layer_i_j + 1 < len(sb_all_layer_outputs[sb_layer_i]):
                        synops_per_sample = synops_per_sample + (sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].sum() / T) * (
                                    sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(-1) + sb_all_layer_outputs[sb_layer_i][sb_layer_i_j+1].size(-1))
                        assert sb_all_layer_outputs[sb_layer_i][sb_layer_i_j].size(0) == T
            # print(sample_firing_rate)
            # print(f"total_spikes: {total_spikes}")
            self.all_samples_synops.append(synops_per_sample.item())

        # if len(id) != 1:
        #     raise ValueError(f"Expected batch size 1 during validation, got {len(id)}")

        # calculate metrics
        mix_y = mix_y.squeeze(0).detach().cpu().numpy()
        ref_y = ref_y.squeeze(0).detach().cpu().numpy()
        est_y = est_y.squeeze(0).detach().cpu().numpy()

        si_sdr = self.si_sdr(est_y, ref_y)
        dns_mos = self.dns_mos(est_y)

        out = si_sdr | dns_mos
        # exit()
        return [out]

    def validation_epoch_end(self, outputs, log_to_tensorboard=True):
        score = 0.0
        if self.compute_firing_rate_per_sample:
            torch.save(torch.tensor(self.all_samples_firing_rate),
                       self.exp_dir / "all_samples_firing_rate.ckpt")
        elif self.compute_firing_rate_per_neuron:
            torch.save(torch.stack(self.all_samples_neurons_firing_rate),
                       self.exp_dir / "all_samples_neurons_firing_rate.ckpt")
        elif self.compute_synops_per_sample:
            torch.save(torch.tensor(self.all_samples_synops),
                       self.exp_dir / "all_samples_synops.ckpt")
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.state.epochs_trained} for dataloader {dataloader_idx}...")

            loss_dict_list = []
            for step_loss_dict_list in tqdm(dataloader_outputs):
                loss_dict_list.extend(step_loss_dict_list)

            df_metrics = pd.DataFrame(loss_dict_list)

            # Compute mean of all metrics
            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            time_now = self._get_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
            score += df_metrics_mean[self.north_star_metric]

            if log_to_tensorboard:
                for metric, value in df_metrics_mean.items():
                    self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.state.epochs_trained)

        return score

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs, log_to_tensorboard=True):
        return self.validation_epoch_end(outputs, log_to_tensorboard=False)
