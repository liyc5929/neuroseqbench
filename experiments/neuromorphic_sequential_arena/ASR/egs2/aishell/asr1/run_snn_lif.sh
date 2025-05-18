#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on : /home/chenxiangma/projects/espnet_1/espnet/tools/miniconda/envs/espnet
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands', params 17.98 M
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/tuning/SNN/train_asr_lif.yaml
inference_config=conf/decode_asr_transformer.yaml

expdir=exp/RNN
asr_tag=SLSTM_LIF_20Epoch_SDBP_decay05_thresh05_lr1

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nj 16 \
    --inference_nj 16 \
    --ngpu 2 \
    --stage 10 \
    --expdir "${expdir}" \
    --asr_tag "${asr_tag}" \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
