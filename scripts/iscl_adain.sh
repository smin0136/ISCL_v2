docker run --gpus '"device=0"' --rm -v /home:/home -ti smin/iscl:2.5.0 python \
	                  /home/Alexandrite/smin/cycle_git/iscl_adain.py  \
			               --output_date='0207' --dir_num='3'  \
				       --adversarial_loss_mode='lsgan' \
			               --epochs=200 \

