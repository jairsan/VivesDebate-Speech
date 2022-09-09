
WAVDIR=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono/
ALIGN_DIR=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/aligned_adus/



for set in "train" "dev";
do
  OUTF=$set
  mkdir -p $OUTF/wavs $OUTF/trans

  if [[ $set == "train" ]];
  then
    vids=$(seq 1 22)
  else
    vids=$(seq 24 26)
  fi

  for fil in $vids;
  do
      ln -s $WAVDIR/Debate$fil.wav $OUTF/wavs/Debate$fil.wav
      cp $ALIGN_DIR/Debate$fil/align.json $OUTF/trans/Debate$fil.align.json
  done
done
