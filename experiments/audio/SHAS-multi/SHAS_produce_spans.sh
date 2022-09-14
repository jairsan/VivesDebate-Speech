TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/

for set in train;
#for set in train dev test;
do

    for maxlen in 5 10 20 30;
    do
        out_path=$PWD/spans/$set.maxlen"$maxlen"/
        mkdir -p $out_path
        path_to_custom_segmentation_yaml=$PWD/infer/$set.maxlen"$maxlen"/segmentation.yaml

        #python3 ../../../src/convert_audio_segmentation_to_spans.py --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path

        if [[ $set == "train" ]];
        then
          for minlen in 1.0 2.0;
          do
            out_filtered=$PWD/spans.minlen$minlen/$set.maxlen"$maxlen"/
            mkdir -p $out_filtered
            python3 ../../../src/filter_spans_by_min_length.py --input_folder $out_path \
              --timestamps_folder $TIMESTAMPS_FOLDER \
              --output_folder $out_filtered \
              --span_min_len $minlen
          done
        fi
    done

done

