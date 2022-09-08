#conda activate shas
AUDIO_LOC=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono_trim
SHAS_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/software/SHAS
out_folder=$PWD/infer/

run=run1
ckpt=step-7272.pt

for set in dev;
do

    path_to_wavs=$AUDIO_LOC/$set/

    for maxlen in 5 10 20;
    do
        out_path=$out_folder/$set.maxlen"$maxlen"/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        thres=0.3

        python3 ${SHAS_ROOT}/src/supervised_hybrid/segment.py \
          -wavs $path_to_wavs \
          -yaml $path_to_custom_segmentation_yaml \
          --dac_threshol $thres \
          -ckpt experiments/tv3_crawl/model_out/$run/ckpts/$ckpt \
          -max $maxlen

        done
done

