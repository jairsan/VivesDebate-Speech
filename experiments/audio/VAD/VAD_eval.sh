AUDIO_LOC=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono_trim
SHAS_ROOT=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/software/SHAS
TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer/

for set in dev test;
do

    path_to_wavs=$AUDIO_LOC/$set/

    for frame_length in 10 20 30;
    do
        for aggressiveness_mode in 1 2 3;
        do
        out_path=$out_folder/$set.length"$frame_length"_agress"$aggressiveness_mode"/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        python3 ../../../src/eval/convert_audio_segmentation_to_labels.py --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path
        done
    done
done

