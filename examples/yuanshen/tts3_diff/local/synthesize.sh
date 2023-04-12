#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
stage=1
stop_stage=1

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize.py \
        --am=diffspeech_aishell3 \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_aishell3 \
        --voc_config=pwg_baker_ckpt_0.4/pwg_default.yaml \
        --voc_ckpt=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
        --voc_stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/test \
        --phones_dict=dump/phone_id_map.txt \
	    --speech_stretchs=dump/train/speech_stretchs.npy \
        --speaker_dict=dump/speaker_id_map.txt
        
fi

# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize.py \
        --am=diffspeech_aishell3 \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_aishell3 \
        --voc_config=hifigan_yuanshen_v1/default.yaml \
        --voc_ckpt=hifigan_yuanshen_v1/snapshot_iter_630000.pdz \
        --voc_stat=hifigan_yuanshen_v1/feats_stats.npy \
        --test_metadata=./metadata.jsonl \
        --output_dir=./tttest_diffspeech_diff_48204 \
        --phones_dict=dump/phone_id_map.txt \
	    --speech_stretchs=dump/train/speech_stretchs.npy \
        --speaker_dict=dump/speaker_id_map.txt
        
fi

# # hifigan
# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     FLAGS_allocator_strategy=naive_best_fit \
#     FLAGS_fraction_of_gpu_memory_to_use=0.01 \
#     python3 ${BIN_DIR}/../synthesize.py \
#         --am=diffspeech_aishell3 \
#         --am_config=conf/default.yaml \
#         --am_ckpt=exp/default/checkpoints/ \
#         --am_stat=dump/train/speech_stats.npy \
#         --voc=hifigan_aishell3 \
#         --voc_config=hifigan_yuanshen_v1/default.yaml \
#         --voc_ckpt=hifigan_yuanshen_v1/snapshot_iter_630000.pdz \
#         --voc_stat=hifigan_yuanshen_v1/feats_stats.npy \
#         --test_metadata=dump/test/norm/metadata.jsonl \
#         --output_dir=./tttest_diffspeech_diff \
#         --phones_dict=dump/phone_id_map.txt \
# 	    --speech_stretchs=dump/train/speech_stretchs.npy \
#         --speaker_dict=dump/speaker_id_map.txt
        
# fi


# hifigan
# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     FLAGS_allocator_strategy=naive_best_fit \
#     FLAGS_fraction_of_gpu_memory_to_use=0.01 \
#     python3 ${BIN_DIR}/../synthesize.py \
#         --am=fastspeech2_aishell3 \
#         --am_config=fastspeech2_yuanshen_new2_66spk2_v1/default.yaml \
#         --am_ckpt=fastspeech2_yuanshen_new2_66spk2_v1/snapshot_iter_20600.pdz \
#         --am_stat=fastspeech2_yuanshen_new2_66spk2_v1/speech_stats.npy \
#         --voc=hifigan_aishell3 \
#         --voc_config=hifigan_yuanshen_v1/default.yaml \
#         --voc_ckpt=hifigan_yuanshen_v1/snapshot_iter_630000.pdz \
#         --voc_stat=hifigan_yuanshen_v1/feats_stats.npy \
#         --test_metadata=dump/test/norm/metadata.jsonl \
#         --output_dir=./tttest_fs2 \
#         --phones_dict=dump/phone_id_map.txt \
# 	    --speech_stretchs=dump/train/speech_stretchs.npy \
#         --speaker_dict=dump/speaker_id_map.txt
        
# fi

