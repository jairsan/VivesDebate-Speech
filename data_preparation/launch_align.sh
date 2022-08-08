rm -r align_log/

mkdir -p aligned/
mkdir -p align_log/

for i in {1..29};
do

nr=$(printf "%05d" $i)
media=audios/Debate$nr.wav
trans=transcriptions_prepro/Debate$i.txt
out=aligned/Debate$i/

qsubmit -gmem 2.5G -m 8 -a uc3m -o align_log/Debate$i.log ./align2.sh config.bash $media $trans $out -b /home/jjorge/2020_05_TASLP/bin/tlk/build/bin/ -S /home/jiranzo/trabajo/git/mllp-speech-data-filtering

done
