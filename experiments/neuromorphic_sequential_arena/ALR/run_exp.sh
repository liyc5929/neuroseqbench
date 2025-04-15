# SFNN
## LIF
CUDA_VISIBLE_DEVICES=6 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.95 --threshold 0.8 --cos-lr --hidden-size 512 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --learning-rule stbp --final-step-cls --amp --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay095_thresh_08_stbp_512_lastStep   # 18.82 17.67 17.83
## CELIF
CUDA_VISIBLE_DEVICES=4 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.3 --threshold 0.5 --cos-lr --hidden-size 512 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron celif --beta 0.02 --amp --final-step-cls --grad-clip 1.0 --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay03_thresh_05_512_celif_beta002_lastStep_gc1_d # 99.15 47.32 48.32
## LTC
CUDA_VISIBLE_DEVICES=5 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.95 --threshold 0.8 --cos-lr --hidden-size 352 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron ltc --amp --final-step-cls --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay095_thresh_08_352_ltc_lastStep # 53.72 48.46 48.93
## SPSN
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.8 --threshold 0.8 --cos-lr --hidden-size 512 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron spsn --amp --final-step-cls --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay08_thresh_08_512_spsn_lastStep # 95.52 45.17 45.73
## PMSN
CUDA_VISIBLE_DEVICES=2 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.8 --threshold 0.8 --cos-lr --hidden-size 512 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron pmsn  --final-step-cls --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay08_thresh_08_512_pmsn_lastStep # 99.70 56.56 57.43


# SRNN
## LIF
CUDA_VISIBLE_DEVICES=3 python main_train.py --dataset dvslip --net ffsnn --lr 5e-4 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.95 --threshold 0.8 --cos-lr --hidden-size 460 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --learning-rule stbp --recurrent --final-step-cls --amp --name dvslip_T200_tri_bn_L6_bs256_5e-4_decay095_thresh_08_stbp_460_rlif_lastStep # 36.1775 34.0905 34.71
## CELIF
CUDA_VISIBLE_DEVICES=1 python main_train.py --dataset dvslip --net ffsnn --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.3 --threshold 0.5 --cos-lr --hidden-size 460 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron celif --beta 0.02 --amp --final-step-cls --alpha 1.0 --recurrent --grad-clip 1.0 --name dvslip_T200_tri_bn_L6_bs256_3e-3_decay03_thresh_05_460_celif_beta002_lastStep_rec_gc1_BB # 98.27 50.43 51.64
## LTC
CUDA_VISIBLE_DEVICES=1 python main_train.py --dataset dvslip --net ffsnn --lr 5e-4 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.95 --threshold 0.8 --cos-lr --hidden-size 340 --hidden-layers 6  --surrogate triangle --bn bn --dropout 0.0 --neuron ltc --amp --final-step-cls --recurrent --name dvslip_T200_tri_bn_L6_bs256_5e-4_decay095_thresh_08_340_ltc_lastStep_rec_b # 75.53 55.98 56.64

# neural architecture
## GSN
CUDA_VISIBLE_DEVICES=1 python main_train.py --dataset dvslip --net spklstm --lr 5e-4 --optim adam --epochs 100 --batch-size 256 --time-window 200 --hidden-size 256 --hidden-layers 6 --cos-lr  --rnn-type gru --neuron lifnode --amp --threshold 0.1 --amp --final-step-cls --grad-clip 1.0 --name dvslip_T200_triL6_bs256_5e-4_thresh_01_256_GSU_lastStep_gc1 # 19.13 21.17 21.17
## TCN
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset dvslip --net tcn --lr 3e-3 --optim adam --epochs 100 --batch-size 256  --time-window 200   --hidden-size 75 --hidden-layers 6 --neuron lifnode --ksize 7 --threshold 0.5 --amp --final-step-cls --name dvslip_T200_triL6_bs256_3e-3_thresh_05_75_spkTCN_lastStep # 74.41 45.25 47.14
## Spike-Driven Transformer
CUDA_VISIBLE_DEVICES=2 python main_train.py --dataset dvslip --net spktransformer --lr 5e-4 --optim adam --batch-size 256 --epochs 100  --time-window 200 --hidden-size 272  --hidden-layers 6 --surrogate triangle --threshold 0.8 --decay 0.95 --amp --final-step-cls --nhead 4 --name dvslip_T200_triL6_bs256_5e-4_thresh_08_272_spkFormer_lastStep_nhead4 # 40.68 39.62 39.62
## Binary S4D
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset dvslip --net binaryssm --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.8 --threshold 0.8 --cos-lr --hidden-size 435 --hidden-layers 6  --surrogate triangle --neuron pmsn  --final-step-cls --amp --name dvslip_T200_tri_bn_L6_bs256_3e-3_435_binaryssn_lastStep_dp02_wd5e-4 --dropout 0.2 --wd 5e-4 # 96.78 44.58 44.80
## GSU-SSM
CUDA_VISIBLE_DEVICES=1 python main_train.py --dataset dvslip --net gsussm --lr 3e-3 --optim adam --epochs 100 --batch-size 256 --time-window 200 --decay 0.8 --threshold 0.8 --cos-lr --hidden-size 485 --hidden-layers 6  --surrogate triangle --neuron pmsn  --final-step-cls --name dvslip_T200_tri_bn_L6_bs256_3e-3_485_gsussn_lastStep_dp02_wd5e-4 --dropout 0.2 --wd 5e-4 # 97.90 40.76 41.35
