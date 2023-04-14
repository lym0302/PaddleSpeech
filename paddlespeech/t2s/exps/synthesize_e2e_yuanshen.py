# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
from pathlib import Path

import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_sentences_svs
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.exps.syn_utils import voc_to_static
from paddlespeech.t2s.utils import str2bool
import time


def evaluate(args):

    # Init body.
    with open(args['am_config']) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args['voc_config']) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    t_start = time.time()

    # frontend
    frontend = get_frontend(
        lang=args['lang'],
        phones_dict=args['phones_dict'],)
    print("frontend done!")

    # acoustic model
    am_name = args['am'][:args['am'].rindex('_')]
    am_dataset = args['am'][args['am'].rindex('_') + 1:]
    am_inference = get_am_inference(
        am=args['am'],
        am_config=am_config,
        am_ckpt=args['am_ckpt'],
        am_stat=args['am_stat'],
        phones_dict=args['phones_dict'],
        speaker_dict=args['speaker_dict'],
        speech_stretchs=args['speech_stretchs'], )
    print("acoustic model done!")

    # vocoder
    voc_inference = get_voc_inference(
        voc=args['voc'],
        voc_config=voc_config,
        voc_ckpt=args['voc_ckpt'],
        voc_stat=args['voc_stat'])
    print("voc done!")
    
    merge_sentences = False
    # front    
    frontend_dict = run_frontend(
        frontend=frontend,
        text=args['text'],
        merge_sentences=merge_sentences,
        lang=args['lang'],)
    
    phone_ids = frontend_dict['phone_ids']
    with paddle.no_grad():
        flags = 0
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i]
            
            # am
            if am_dataset in {"aishell3", "vctk", "mix", "canton"}:
                spk_id = paddle.to_tensor(args['spk_id'])
                mel = am_inference(part_phone_ids, spk_id)
            else:
                mel = am_inference(part_phone_ids)
        
            # voc
            wav = voc_inference(mel)
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
            wav = wav_all.numpy()
    
    sf.write(
            str(args['output']), wav, samplerate=am_config.fs)
    
    t_end = time.time()

    rtf = (t_end - t_start) / (wav.size / am_config.fs)
    print(f"audio dur: {(wav.size / am_config.fs)}, infer spend time: {t_end - t_start}, RTF: {rtf}")



def main():
    args = {}
    args['am'] = 'diffspeech_aishell3'
    args['am_config'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/default.yaml'
    args['am_ckpt'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/snapshot_iter_48204.pdz'
    args['am_stat'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/speech_stats.npy'
    args['phones_dict'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/phone_id_map.txt'
    args['speaker_dict'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/speaker_id_map.txt'
    args['speech_stretchs'] = 'diffspeech_yuanshen_new2_66spks_1.5.0/speech_stretchs.npy'
    
    args['voc'] = 'hifigan_aishell3'
    args['voc_config'] = 'hifigan_yuanshen_v1/default.yaml'
    args['voc_ckpt'] = 'hifigan_yuanshen_v1/snapshot_iter_630000.pdz'
    args['voc_stat'] = 'hifigan_yuanshen_v1/feats_stats.npy'
    
    args['lang'] = 'zh'
    args["ngpu"] = 1
    args['spk_id'] = 0
    args['text'] = "克哈，是我们安全的港湾。"
    args['output'] = "./llym/aa.wav"
    

    if args['ngpu'] == 0:
        paddle.set_device("cpu")
    elif args['ngpu'] > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
