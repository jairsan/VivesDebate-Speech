#TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer

#for set in dev test;
for set in dev;
do

    #for frame_length in 10 20 30;
    for frame_length in 10;
    do
        #for aggressiveness_mode in 1 2 3;
        for aggresiveness_mode in 1;

        do
        out_path=$out_folder/$set.length"$frame_length"_agress"$aggresiveness_mode"
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        # Without segment classifier
        python3 ../../../src/convert_audio_segmentation_to_labels.py --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path
        
        # With segment classifier
        #python3 ../../../src/convert_audio_segmentation_to_labels.py --segment_classifier ../segment_classifier/model.pkl --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path

        echo "##########"
        echo "frame_length$frame_length.aggressiveness_mode$aggresiveness_mode"
        if [[ $set == "dev" ]];
        then
            python3 ../../../src/eval/eval.py --hypotheses_files $out_path/Debate24.labels $out_path/Debate25.labels $out_path/Debate26.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt
        else
            python3 ../../../src/eval/eval.py --hypotheses_files $out_path/Debate27.labels $out_path/Debate28.labels $out_path/Debate29.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate27.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate28.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate29.txt
        fi
        echo "##########"

        done
    done
done

