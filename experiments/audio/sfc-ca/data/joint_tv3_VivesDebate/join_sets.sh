TV3_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/experiments/audio/sfc-ca/data/TV3_CRAWL/
VIVES_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/experiments/audio/sfc-ca/data/VivesDebate

tv3_train_set=train
vives_train_set=train
vives_eval_set=dev


mkdir -p train/trans train/wavs
mkdir -p dev/trans dev/wavs

for file in $(ls $TV3_ROOT/$tv3_train_set/trans);
do
  filename="${file%.*.*}"
  ln -s $TV3_ROOT/$tv3_train_set/wavs/$filename.wav train/wavs/
  ln -s $TV3_ROOT/$tv3_train_set/trans/$filename.align.json train/trans/

done

for file in $(ls $VIVES_ROOT/$vives_train_set/trans);
do
  filename="${file%.*.*}"
  ln -s $TV3_ROOT/vives_train_set/wavs/$filename.wav train/wavs/
  ln -s $TV3_ROOT/vives_train_set/trans/$filename.align.json train/trans/

done

for file in $(ls $VIVES_ROOT/$vives_eval_set/trans);
do
  filename="${file%.*.*}"
  ln -s $TV3_ROOT/vives_eval_set/wavs/$filename.wav dev/wavs/
  ln -s $TV3_ROOT/vives_eval_set/trans/$filename.align.json dev/trans/

done