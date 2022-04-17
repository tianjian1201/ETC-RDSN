# EDSR baseline model (x1)
#python main.py --model EDSR --scale 1 --patch_size 48 --batch_size 5 --save edsr_baseline_x1 --epochs 10 --reset --data_train Set5 --data_test Set5 --data_range 1-5/1-5 --save_gt --save_results
#python main.py --data_test Set5+B100+Set14+Urban100 --scale 1 --model EDSR --pre_train ../experiment/edsr_baseline_x1/model/model_best.pt --test_only --save_results

# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download



# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
#python main.py --scale 2 --save RDCN_x2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --data_range 1-800/801-805 --chop --load RDCN_x2 --decay 200-400-600-800-1000 --pre_train ../experiment/RDCN_x2/model/model_best.pt
#python main.py --scale 4 --save RDCNG_x3 --model RDN --epochs 1000 --batch_size 16 --patch_size 63 --data_range 1-800/801-805 --chop --decay 200-400-600-800-1000 --n_colors 1 --pre_train ../experiment/RDCNG_x3/model/model_latest.pt --load RDCNG_x3
#python main.py --scale 3 --save YRDSN_x3 --model RDN --epochs 1000 --batch_size 16 --patch_size 63 --data_range 1-800/801-805 --decay 200-400-600-800-1000 --pre_train ../experiment/YRDSN_x3/model/model_latest.pt --load YRDSN_x3
#python main.py --scale  4 --save RDCN_MIX_x4 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --data_range 1-800/801-805 --decay 200-250-500-750-1000 --chop --pre_train ../experiment/RDCN_MIX_x4/model/model_latest.pt --load RDCN_MIX_x4 --save_models
#python main.py --data_test Set5 --scale 2 --model RDN --pre_train ../experiment/RDCN_v5_x2/model/model_885.pt --test_only --save_results
#python main.py --data_test Set5 --scale 3 --model RDN --pre_train ../experiment/RDCN_v5_x3/model/model_891.pt --test_only --save_results
#python main.py --data_test Set5 --scale 4 --model RDN --pre_train ../experiment/RDCN_v5_x4/model/model_latest.pt --test_only --save_results

#python main.py --scale 2 --save RDN_D16C8G64_Ablationx2 --model RDN --epochs 200 --batch_size 16 --patch_size 64 --reset

#RDCN color img
#python main.py --data_test Set5 --scale 2 --model RDN --pre_train ../experiment/RDN_Test_x2/model/model_1010.pt --test_only --save_results
#python main.py --data_test Set5 --scale 3 --model RDN --pre_train ../experiment/RDN_Test_x3/model/model_best.pt --test_only --save_results
#python main.py --data_test Set5 --scale 4 --model RDN --pre_train ../experiment/RDN_Test_x4/model/model_latest.pt --test_only --save_results
#RDSN color img
#python main.py --data_test Set14 --scale 2 --model RDN --pre_train ../experiment/RDCN_v5_x2/model/model_885.pt --test_only --save_results
#python main.py --data_test Set5+Set14+B100+Urban100+test_image --scale 2 --model RDN --pre_train ../experiment/RDCN_v5_x2/model/model_885.pt --test_only --save_results
#python main.py --data_test Set5+Set14+B100+Urban100+test_image --scale 3 --model RDN --pre_train ../experiment/RDCN_v5_x3/model/model_891.pt --test_only --save_results
#python main.py --data_test Set5+Set14+B100+Urban100+test_image --scale 4 --model RDN --pre_train ../experiment/RDCN_v5_x4/model/model_latest.pt --test_only --save_results
#python main.py --data_test test_image --scale 2 --model RDN --pre_train ../experiment/RDCN_v5_x2/model/model_885.pt --test_only --save_results
#python main.py --data_test test_image --scale 3 --model RDN --pre_train ../experiment/RDCN_v5_x3/model/model_891.pt --test_only --save_results
#python main.py --data_test test_image --scale 4 --model RDN --pre_train ../experiment/RDCN_v5_x4/model/model_latest.pt --test_only --save_results
#python main.py --data_test Urban100 --scale 4 --model RDN --pre_train ../experiment/RDCN_v5_x4/model/model_latest.pt --test_only --save_results
#python main.py --data_test Set5 --scale 4 --model RDN --pre_train ../experiment/RDCN_v5_x4/model/model_latest.pt --test_only --save_results

#RDSN gray img
#python main.py --data_test Set14 --scale 2 --model RDN --pre_train ../experiment/RDCNG_x2/model/model_best.pt --test_only --save_results --n_colors 1
#python main.py --data_test Set14 --scale 3 --model RDN --pre_train ../experiment/RDCNG_x3/model/model_latest.pt --test_only --save_results --n_colors 1
#python main.py --data_test Set14 --scale 4 --model RDN --pre_train ../experiment/RDCNG_x4/model/model_latest.pt --test_only --save_results --n_colors 1
#YRDSN
#python main.py --data_test Set14+Set5+B100+Urban100 --scale 4 --model RDN --pre_train ../experiment/YRDSN_x4/model/model_latest.pt --test_only --save_results
#python main.py --data_test Set14+Set5+B100+Urban100 --scale 3 --model RDN --pre_train ../experiment/YRDSN_x4/model/model_latest.pt --test_only --save_results
#python main.py --data_test Set14+Set5+B100+Urban100 --scale 2 --model RDN --pre_train ../experiment/YRDSN_x4/model/model_latest.pt --test_only --save_results


#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset





# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../models/RCAN_BIX3.pt
#python main.py --data_test Set5 --scale 3 --model RCAN --pre_train ../models/RCAN_BIX3.pt --test_only --save_results
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

#LR像素映射到HR实验
#python main.py  --scale 2 --model RDN --pre_train ../experiment/RDN_D16C8G64_Ablationx2_E1000_backup/model/model_best.pt --test_only --save_results
#python main.py --data_test Set5 --scale 2 --model RDN --epochs 1000 --batch_size 16 --batch_size 16 --patch_size 64 --save RDN_D16C8G64_Ax2_uni_E1000 --ext sep_reset
# start:2020-12-30 13:00
#python main.py --data_test Set5 --scale 2 --model RDN --epochs 1000 --batch_size 16 --batch_size 16 --patch_size 64 --save RDN_D16C8G64_Ax2_uni_E1000 --pre_train ../experiment/RDN_D16C8G64_Ax2_uni_E1000/model/model_latest.pt --load RDN_D16C8G64_Ax2_uni_E1000

#像素填回--x2
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save RDN_D16C8G64_Ax2_bic_E1000 --data_test Set5 --pre_train ../experiment/RDN_D16C8G64_Ax2_bic_E1000/model/model_latest.pt --load  RDN_D16C8G64_Ax2_bic_E1000
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 8 --patch_size 64 --save RDN_D16C8G64_Ax2_bic_E1000 --data_test Set14+Set5+B100+Urban100 --pre_train ../experiment/RDN_D16C8G64_Ax2_bic_E1000/model/model_latest.pt --load RDN_D16C8G64_Ax2_bic_E1000
#填回优化测试
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save test_20210114

#测试
#python main.py --scale 2 --model RDN --data_test Set14 --pre_train ../experiment/RDN_D16C8G64_Ax2_bic_E1000/model/model_best.pt --test_only --save_results
#去掉全局跳跃连接
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save RDN_D16C8G64_Ax2_noGSC_E1000 --data_test Set5 --pre_train ../experiment/RDN_D16C8G64_Ax2_noGSC_E1000/model/model_latest.pt --load  RDN_D16C8G64_Ax2_noGSC_E1000
#去掉GSC/SM,不填回
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save RDN_D16C8G64_Ax2_noGSC_SM_E1000 --data_test Set5 
#--pre_train ../experiment/RDN_D16C8G64_Ax2_noGSC_SM_E1000/model/model_latest.pt --load  RDN_D16C8G64_Ax2_noGSC_SM_E1000
#python main.py --scale 2 --model RDN --data_test Set14+Set5+B100+Urban100 --pre_train ../experiment/RDN_D16C8G64_Ax2_noGSC_SM_E1000/model/model_best.pt --test_only --save_results
#去掉GSC/SM,填回
#python main.py --scale 2 --model RDN --data_test Set14+Set5+B100+Urban100 --pre_train ../experiment/RDN_D16C8G64_Ax2_noGSC_SM_insert_E1000/model/model_best.pt --test_only --save_results
#调试

#区间内不填回，0填回,迁移训练
#X4
#python main.py --scale 4 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save hr_sr_X4_replace_std_finetuning_20210322 --data_test Set5 --pre_train ../experiment/RDN_D16C8G64_Ax4_bic_E1000/model/model_latest.pt
#python main.py --scale 2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save ablation_NOGSC_NOSM_insert --data_test Set5
#python main.py --scale 4 --model RDN  --pre_train ../experiment/hr_sr_X4_replace_std_finetuning_20210322/model/model_best.pt --test_only --save_results --data_test Set5+Set14+B100+Urban100

#X3
#python main.py --scale 3 --model RDN --epochs 1000 --batch_size 16 --patch_size 63 --save hr_sr_X3_replace_std_finetuning_20210329 --data_test Set5 --pre_train ../experiment/RDN_D16C8G64_Ax3_bic_E1000/model/model_latest.pt


#测试
#python main.py --scale 2 --model RDN  --pre_train ../experiment/ablation_NOGSC_NOSM_insert/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100
#python main.py --scale 3 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax3_bic_E1000/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100
#python main.py --scale 4 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax4_bic_E1000/model/model_best.pt --test_only --save_results --data_test Urban100

# python main.py --scale 4 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save test20210729 --data_test Set5
#python main.py --scale 2 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax2_bic_E1000/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100 --save test

# 测试
# python main.py --scale 2 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax2_bic_E1000/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100
# python main.py --scale 3 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax3_bic_E1000/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100
# python main.py --scale 4 --model RDN  --pre_train ../experiment/RDN_D16C8G64_Ax4_bic_E1000/model/model_best.pt --test_only --save_results --data_test Set14+Set5+B100+Urban100
