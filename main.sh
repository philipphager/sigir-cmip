currentDate=`date +'%s'`

python train.py hydra.job_logging.root.level=WARNING +datetime="$currentDate" $@
python evaluate.py hydra.job_logging.root.level=WARNING +datetime="$currentDate" $@