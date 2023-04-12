# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from espnet(https://github.com/espnet/espnet)
"""DiffSinger related modules for paddle"""
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.models.fastspeech2.fastspeech2 import FastSpeech2
from paddlespeech.t2s.modules.diffnet import DiffNet
from paddlespeech.t2s.modules.diffusion import GaussianDiffusion


class DiffSpeech(nn.Layer):
    """DiffSinger module.

    This is a module of DiffSinger described in `DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism`._
    .. _`DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism`:
        https://arxiv.org/pdf/2105.02446.pdf

    Args:

    Returns:

    """

    def __init__(
            self,
            # min and max spec for stretching before diffusion
            spec_min: paddle.Tensor,
            spec_max: paddle.Tensor,
            # fastspeech2midi config
            idim: int,
            odim: int,
            # use_energy_pred: bool=False,
            # use_postnet: bool=False,
            # # music score related 
            # note_num: int=300,
            # is_slur_num: int=2,
            fastspeech2_params: Dict[str, Any]={
                "adim": 256,
                "aheads": 2,
                "elayers": 4,
                "eunits": 1024,
                "dlayers": 4,
                "dunits": 1024,
                "positionwise_layer_type": "conv1d",
                "positionwise_conv_kernel_size": 1,
                "use_scaled_pos_enc": True,
                "use_batch_norm": True,
                "encoder_normalize_before": True,
                "decoder_normalize_before": True,
                "encoder_concat_after": False,
                "decoder_concat_after": False,
                "reduction_factor": 1,
                # for transformer
                "transformer_enc_dropout_rate": 0.1,
                "transformer_enc_positional_dropout_rate": 0.1,
                "transformer_enc_attn_dropout_rate": 0.1,
                "transformer_dec_dropout_rate": 0.1,
                "transformer_dec_positional_dropout_rate": 0.1,
                "transformer_dec_attn_dropout_rate": 0.1,
                "transformer_activation_type": "gelu",
                # duration predictor
                "duration_predictor_layers": 2,
                "duration_predictor_chans": 384,
                "duration_predictor_kernel_size": 3,
                "duration_predictor_dropout_rate": 0.1,
                # pitch predictor
                "use_pitch_embed": True,
                "pitch_predictor_layers": 2,
                "pitch_predictor_chans": 384,
                "pitch_predictor_kernel_size": 3,
                "pitch_predictor_dropout": 0.5,
                "pitch_embed_kernel_size": 9,
                "pitch_embed_dropout": 0.5,
                "stop_gradient_from_pitch_predictor": False,
                # energy predictor
                "use_energy_embed": False,
                "energy_predictor_layers": 2,
                "energy_predictor_chans": 384,
                "energy_predictor_kernel_size": 3,
                "energy_predictor_dropout": 0.5,
                "energy_embed_kernel_size": 9,
                "energy_embed_dropout": 0.5,
                "stop_gradient_from_energy_predictor": False,
                # postnet
                "postnet_layers": 5,
                "postnet_chans": 512,
                "postnet_filts": 5,
                "postnet_dropout_rate": 0.5,
                # spk emb
                "spk_num": None,
                "spk_embed_dim": None,
                "spk_embed_integration_type": "add",
                # training related
                "init_type": "xavier_uniform",
                "init_enc_alpha": 1.0,
                "init_dec_alpha": 1.0,
                # speaker classifier
                "enable_speaker_classifier": False,
                "hidden_sc_dim": 256,
            },
            # denoiser config
            denoiser_params: Dict[str, Any]={
                "in_channels": 80,
                "out_channels": 80,
                "kernel_size": 3,
                "layers": 20,
                "stacks": 5,
                "residual_channels": 256,
                "gate_channels": 512,
                "skip_channels": 256,
                "aux_channels": 256,
                "dropout": 0.,
                "bias": True,
                "use_weight_norm": False,
                "init_type": "kaiming_normal",
            },
            # diffusion config
            diffusion_params: Dict[str, Any]={
                "num_train_timesteps": 100,
                "beta_start": 0.0001,
                "beta_end": 0.06,
                "beta_schedule": "squaredcos_cap_v2",
                "num_max_timesteps": 60,
                "stretch": True,
            }, ):
        """Initialize DiffSinger module.

        Args:
            spec_min (paddle.Tensor): The minimum value of the feature(mel) to stretch before diffusion.
            spec_max (paddle.Tensor): The maximum value of the feature(mel) to stretch before diffusion.
            idim (int): Dimension of the inputs (Input vocabrary size.).
            odim (int): Dimension of the outputs (Acoustic feature dimension.).
            use_energy_pred (bool, optional): whether use energy predictor. Defaults False.
            use_postnet (bool, optional): whether use postnet. Defaults False.
            note_num (int, optional): The number of note. Defaults to 300.
            is_slur_num (int, optional): The number of slur. Defaults to 2.
            fastspeech2_params (Dict[str, Any]): Parameter dict for fastspeech2 module.
            denoiser_params (Dict[str, Any]): Parameter dict for dinoiser module.
            diffusion_params (Dict[str, Any]): Parameter dict for diffusion module.
        """
        assert check_argument_types()
        super().__init__()
        self.fs2 = FastSpeech2(
            idim=idim,
            odim=odim,
            **fastspeech2_params,)
        denoiser = DiffNet(**denoiser_params)
        self.diffusion = GaussianDiffusion(
            denoiser,
            **diffusion_params,
            min_values=spec_min,
            max_values=spec_max,)

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            durations: paddle.Tensor,
            pitch: paddle.Tensor,
            energy: paddle.Tensor,
            spk_emb: paddle.Tensor=None,
            spk_id: paddle.Tensor=None,
            only_train_fs2: bool=True,
    ) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor], paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            text(Tensor(int64)): 
                Batch of padded token (phone) ids (B, Tmax).
            note(Tensor(int64)): 
                Batch of padded note (element in music score) ids (B, Tmax).
            note_dur(Tensor(float32)): 
                Batch of padded note durations in seconds (element in music score) (B, Tmax).
            is_slur(Tensor(int64)): 
                Batch of padded slur (element in music score) ids (B, Tmax).
            text_lengths(Tensor(int64)): 
                Batch of phone lengths of each input (B,).
            speech(Tensor[float32]): 
                Batch of padded target features (e.g. mel) (B, Lmax, odim).
            speech_lengths(Tensor(int64)): 
                Batch of the lengths of each target features (B,).
            durations(Tensor(int64)): 
                Batch of padded token durations in frame (B, Tmax).
            pitch(Tensor[float32]): 
                Batch of padded frame-averaged pitch (B, Lmax, 1).
            energy(Tensor[float32]): 
                Batch of padded frame-averaged energy (B, Lmax, 1).
            spk_emb(Tensor[float32], optional): 
                Batch of speaker embeddings (B, spk_embed_dim).
            spk_id(Tnesor[int64], optional(int64)): 
                Batch of speaker ids (B,)
            only_train_fs2(bool):
                Whether to train only the fastspeech2 module

        Returns:

        """
        # only train fastspeech2 module firstly
        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits = self.fs2(
            text=text,
            text_lengths=text_lengths,
            speech=speech,
            speech_lengths=speech_lengths,
            durations=durations,
            pitch=pitch,
            energy=energy,
            spk_id=spk_id,
            spk_emb=spk_emb)
        if only_train_fs2:
            return before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, spk_logits

        # get the encoder output from fastspeech2 as the condition of denoiser module
        cond_fs2, mel_masks = self.fs2.encoder_infer_batch(
            text=text,
            text_lengths=text_lengths,
            speech_lengths=speech_lengths,
            ds=durations,
            ps=pitch,
            es=energy,
            spk_id=spk_id,
            spk_emb=spk_emb)
        cond_fs2 = cond_fs2.transpose((0, 2, 1))

        # get the output(final mel) from diffusion module
        noise_pred, noise_target = self.diffusion(
            speech.transpose((0, 2, 1)), cond_fs2)
        return noise_pred, noise_target, mel_masks

    def inference(
            self,
            text: paddle.Tensor,
            spk_emb=None,
            spk_id=None,
            get_mel_fs2: bool=False, ):
        """Run inference

        Args:
            text(Tensor(int64)): 
                Batch of padded token (phone) ids (B, Tmax).
            note(Tensor(int64)): 
                Batch of padded note (element in music score) ids (B, Tmax).
            note_dur(Tensor(float32)): 
                Batch of padded note durations in seconds (element in music score) (B, Tmax).
            is_slur(Tensor(int64)): 
                Batch of padded slur (element in music score) ids (B, Tmax).
            get_mel_fs2 (bool, optional): . Defaults to False.
                Whether to get mel from fastspeech2 module.

        Returns:
            
        """
        mel_fs2, _, _, _ = self.fs2.inference(text=text, spk_emb=spk_emb, spk_id=spk_id,)
        if get_mel_fs2:
            return mel_fs2
        mel_fs2 = mel_fs2.unsqueeze(0).transpose((0, 2, 1))
        cond_fs2 = self.fs2.encoder_infer(text=text, spk_emb=spk_emb, spk_id=spk_id,)
        cond_fs2 = cond_fs2.transpose((0, 2, 1))
        noise = paddle.randn(mel_fs2.shape)
        mel = self.diffusion.inference(
            noise=noise,
            cond=cond_fs2,
            ref_x=mel_fs2,
            scheduler_type="ddpm",
            num_inference_steps=60)
        mel = mel.transpose((0, 2, 1))
        return mel[0]


class DiffSpeechInference(nn.Layer):
    def __init__(self, normalizer, model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = model

    def forward(self, text, spk_id=None, spk_emb=None, get_mel_fs2: bool=False):
        """Calculate forward propagation.

        Args:
            text(Tensor(int64)): 
                Batch of padded token (phone) ids (B, Tmax).
            note(Tensor(int64)): 
                Batch of padded note (element in music score) ids (B, Tmax).
            note_dur(Tensor(float32)): 
                Batch of padded note durations in seconds (element in music score) (B, Tmax).
            is_slur(Tensor(int64)): 
                Batch of padded slur (element in music score) ids (B, Tmax).
            get_mel_fs2 (bool, optional): . Defaults to False.
                Whether to get mel from fastspeech2 module.

        Returns:
            logmel(Tensor(float32)): denorm logmel, [T, mel_bin]
        """
        normalized_mel = self.acoustic_model.inference(
            text=text, spk_id=spk_id, spk_emb=spk_emb,
            get_mel_fs2=get_mel_fs2)
        # logmel = normalized_mel
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel
