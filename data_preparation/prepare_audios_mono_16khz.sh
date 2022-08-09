outfolder=audios_16khz_mono

mkdir -p $outfolder

for i in {1..29};
do
    nr=$(printf "%05d" $i)
    ffmpeg -i audios/Debate$nr.wav -ac 1 -ar 16000 -hide_banner -loglevel error $outfolder/Debate$i.wav
done
