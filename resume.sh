model_dir="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/msg_models/exp_1"
images_dir="/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/van_gogh_prep"

resume_epoch=730
epoch_limit=1000

python3 sourcecode/train.py --depth=6 \
                            --latent_size=512 \
                            --start $resume_epoch \
                            --num_epochs=$epoch_limit \
                            --batch_size=5 \
                            --feedback_factor=1 \
                            --checkpoint_factor=10 \
                            --flip_augment=True \
                            --sample_dir=samples/exp_1 \
                            --model_dir=$model_dir \
                            --images_dir=$images_dir \
                            --generator_file=$model_dir/GAN_GEN_$resume_epoch.pth \
                            --generator_optim_file=$model_dir/GAN_GEN_OPTIM_$resume_epoch.pth \
                            --shadow_generator_file=$model_dir/GAN_GEN_SHADOW_$resume_epoch.pth \
                            --discriminator_file=$model_dir/GAN_DIS_$resume_epoch.pth \
                            --discriminator_optim_file=$model_dir/GAN_DIS_OPTIM_$resume_epoch.pth
