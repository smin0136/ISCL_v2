docker run --gpus '"device=0"' --rm -v /home:/home -ti smin/iscl:2.5.0 python \
	                  /home/Alexandrite/smin/scripts/ISCL_v2/pre_adain.py  \
			               --output_date='0208' --dir_num='1'  \
				            --adversarial_loss_mode='lsgan' \
			               --epochs=200 \

