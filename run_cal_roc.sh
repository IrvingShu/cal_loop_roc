nohup python -u ./src/cal_roc.py \
    --score-list-path=./score.txt \
    --roc-save-txt=./roc_save.txt > ./logs/run.log 2>&1 &
