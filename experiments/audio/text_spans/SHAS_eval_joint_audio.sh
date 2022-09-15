#TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer/
#for set in dev test;
for set in dev;
do

        out_path=$out_folder/$set/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml

        classifier=audio_classifier_table:170

        python3 ../../../src/convert_audio_segmentation_to_labels.py --segment_classifier audio-transformers:../segment_classifier/$classifier --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path
        echo "##########"
        if [[ $set == "dev" ]];
        then
            python3 ../../../src/eval/eval.py --convert_to_bio --hypotheses_files $out_path/Debate24.labels $out_path/Debate25.labels $out_path/Debate26.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt
        else
            python3 ../../../src/eval/eval.py --convert_to_bio --hypotheses_files $out_path/Debate27.labels $out_path/Debate28.labels $out_path/Debate29.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate27.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate28.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate29.txt
        fi
        echo "##########"

        done

