AUDIO_LOC=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/audios_16khz_mono_trim
REFERENCES_LOC=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer/

for set in dev test;
do
    path_to_wavs=$AUDIO_LOC/$set/

    for maxlen in 10 20 30;
    do
        out_path=$out_folder/$set.maxlen"$maxlen"/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        python3 ../../../src/segment_audio_oracle.py \
          --reference_files_location $REFERENCES_LOC \
          --wavs $path_to_wavs \
          --yaml "$path_to_custom_segmentation_yaml" \
          --max_len $maxlen

        done
done