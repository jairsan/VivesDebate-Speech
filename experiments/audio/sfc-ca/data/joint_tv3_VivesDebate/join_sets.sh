TV3_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/experiments/audio/sfc-ca/data/TV3_CRAWL/
VIVES_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/experiments/audio/sfc-ca/data/VivesDebate

tv3_train_set=train
vives_train_set=train
vives_eval_set=dev

rm -r train/ dev/

mkdir -p train/trans train/wavs
mkdir -p dev/trans dev/wavs

rm train.lst dev.lst

for file in $(ls $TV3_ROOT/$tv3_train_set/trans);
do
  filename="${file%.*.*}"
  ln -s $TV3_ROOT/$tv3_train_set/wavs/$filename.wav train/wavs/
  ln -s $TV3_ROOT/$tv3_train_set/trans/$filename.align.json train/trans/
  echo $filename >> train.lst
done

for file in $(ls $VIVES_ROOT/$vives_train_set/trans);
do
  filename="${file%.*.*}"
  ln -s $VIVES_ROOT/$vives_train_set/wavs/$filename.wav train/wavs/
  ln -s $VIVES_ROOT/$vives_train_set/trans/$filename.align.json train/trans/
  echo $filename >> train.lst
done

for file in $(ls $VIVES_ROOT/$vives_eval_set/trans);
do
  filename="${file%.*.*}"
  ln -s $VIVES_ROOT/$vives_eval_set/wavs/$filename.wav dev/wavs/
  ln -s $VIVES_ROOT/$vives_eval_set/trans/$filename.align.json dev/trans/
  echo $filename >> dev.lst
done

python3 generate_yaml.py train/trans/ train.lst train/train.yaml
python3 generate_yaml.py dev/trans/ dev.lst dev/dev.yaml
