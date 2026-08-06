#! /bin/bash

# ucf
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ucfcrime --feature-size 512 --emb_folder sent_emb_n --rgb_list list/ucf-clip-train.list --test_rgb_list list/ucf-clip-test.list --aggregate_text --batch_size 32 --max_epoch 1000 --margin_prompt 50 --_lambda 0.03 --feedback_aug 1 --omega_v 1.0 --omega_t 0.01 --decoders_mode geu --feedback_mode reweight --depth 2 --fusion concat --prompt_num 28 --lr [0.0002]*15000

# uca
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ucfcrime --feature-size 512 --emb_folder uca_sent_emb_n --rgb_list list/ucf-clip-train.list --test_rgb_list list/ucf-clip-test.list --aggregate_text --batch_size 32 --max_epoch 1000 --margin_prompt 100 --_lambda 0.05 --feedback_aug 1 --omega_v 0.5 --omega_t 0.5 --decoders_mode geu --feedback_mode concat --depth 2 --fusion concat --prompt_num 28 --lr [0.0002]*15000

# xd
CUDA_VISIBLE_DEVICES=0 python main.py --dataset violence --feature-size 512 --emb_folder sent_emb_n --rgb_list list/violence-clip.list --test_rgb_list list/violence-clip-test.list --aggregate_text --batch_size 32 --max_epoch 1000 --margin_prompt 50 --_lambda 0.01 --feedback_aug 1 --omega_v 1.0 --omega_t 0.3 --decoders_mode geu --feedback_mode concat --depth 2 --fusion add --prompt_num 14 --lr [0.0001]*15000

# 93.65
CUDA_VISIBLE_DEVICES=0 python main.py --dataset tad --feature-size 512 --emb_folder sent_emb_n --rgb_list list/tad-clip-train.list --test_rgb_list list/tad-clip-test.list --aggregate_text --batch_size 32 --max_epoch 1000 --margin_prompt 50 --_lambda 0.003 --feedback_aug 1 --omega_v 1.0 --omega_t 0.6 --decoders_mode geu --feedback_mode reweight --depth 2 --fusion concat --prompt_num 10 --lr [0.0005]*15000


