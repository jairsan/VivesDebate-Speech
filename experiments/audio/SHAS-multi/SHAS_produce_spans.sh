#TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/

for set in dev test;
do

    for maxlen in 5 10 20 30;
    do
        out_path=$PWD/spans/$set.maxlen"$maxlen"/
        mkdir -p $out_path
        path_to_custom_segmentation_yaml=$PWD/infer/$set.maxlen"$maxlen"/segmentation.yaml

        python3 ../../../src/convert_audio_segmentation_to_spans.py --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path

    done

done

