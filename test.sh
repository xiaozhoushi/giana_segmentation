#python3 evaluation.py --data_path /mnt/bd/aurora-mtrc-data/data/giana/segmentation/test/hdtest --model_upsample_num 5 --result_save_path ./result --model_path ./checkpoint/segformer/20210922_184747/checkpoint.pth.tar --model segformer 

python3 evaluation.py --data_path /mnt/bd/aurora-mtrc-data/data/miccai_seg/test/cvc612 --model_upsample_num 5 --result_save_path ./hight --model_path ./checkpoint/segformer/20220121_111959/model_best.pth.tar --model segformer --crop_black true
