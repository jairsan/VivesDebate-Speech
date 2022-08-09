rm -r align_log_adus/

mkdir -p aligned_adus/
mkdir -p align_log_adus/

for i in {1..29};
do

nr=$(printf "%05d" $i)
media=audios/Debate$nr.wav
trans=adus_text_prepro/Debate$i.txt
out=aligned_adus/Debate$i/
log=align_log_adus/Debate$i.log

rm -r $out/
rm $log
qsubmit -gmem 2.5G -m 8 -a uc3m -o $log ./align2.sh config.bash $media $trans $out -b /home/jjorge/2020_05_TASLP/bin/tlk/build/bin/ -S /home/jiranzo/trabajo/git/mllp-speech-data-filtering


done
