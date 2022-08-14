AUDIO_LOC=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono_trim
SHAS_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/software/SHAS
out_folder=$PWD/infer/

for set in dev test;
do

    path_to_wavs=$AUDIO_LOC/$set/

    for frame_length in 10 20 30;
    do
        for aggressiveness_mode in 1 2 3;
        do
        path_to_custom_segmentation_yaml=$out_folder/$set.length"$frame_length"_agress"$aggressiveness_mode"/segmentation.yaml

        python3 ${SHAS_ROOT}/src/segmentation_methods/pause_based.py \
          -wavs $path_to_wavs \
          -yaml $path_to_custom_segmentation_yaml \
          -l $frame_length \
          -a $aggressiveness_mode

        python3 convert_segmentation_to_labels.py $path_to_custom_segmentation_yaml
        done
    done
done

