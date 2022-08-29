#TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer/

for set in dev test;
do

    for frame_length in 10 20 30;
    do
        for aggressiveness_mode in 1 2 3;
        do
        out_path=$out_folder/$set.length"$frame_length"_agress"$aggressiveness_mode"/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        python3 ../../../src/eval/convert_audio_segmentation_to_labels.py --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path

        if [[ $set == "dev" ]];
        then
          debates="24 25 26"
        else
          debates="27 28 29"
        fi

        for debate in $debates;
        do
          echo "##########"
          echo "frame_length$frame_length.aggressiveness_mode$aggressiveness_mode.debate$debate"
          python3 ../../../src/eval/eval.py --hypotheses_files $out_path/Debate$debate.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate$debate.txt
          echo "##########"
        done
    done
done
done

