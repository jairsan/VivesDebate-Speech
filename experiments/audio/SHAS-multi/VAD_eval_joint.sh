#TIMESTAMPS_FOLDER=/scratch/jiranzotmp/trabajo/ICASSP2023_argumentation/data_preparation/DATA/BIO_arg_timestamps/
TIMESTAMPS_FOLDER=../../../data_preparation/DATA/BIO_arg_timestamps/
out_folder=$PWD/infer/

#for set in dev test;
for set in dev;
do

    #for maxlen in 10 20 30;
    for maxlen in 20;
    do
        out_path=$out_folder/$set.maxlen"$maxlen"/
        path_to_custom_segmentation_yaml=$out_path/segmentation.yaml
        #--segment_classifier ../segment_classifier/model.pkl
        #--segment_classifier transformers#../segment_classifier/tmp_transformers_models/checkpoint-125/
        python3 ../../../src/convert_audio_segmentation_to_labels.py --segment_classifier transformers#../segment_classifier/tmp_transformers_models/checkpoint-250/ --yaml_file $path_to_custom_segmentation_yaml --timestamps_folder $TIMESTAMPS_FOLDER --output_folder $out_path

        echo "##########"
        echo "maxlen$maxlen"
        if [[ $set == "dev" ]];
        then
            python3 ../../../src/eval/eval.py --convert_to_bio --hypotheses_files $out_path/Debate24.labels $out_path/Debate25.labels $out_path/Debate26.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt
        else
            python3 ../../../src/eval/eval.py --convert_to_bio --hypotheses_files $out_path/Debate27.labels $out_path/Debate28.labels $out_path/Debate29.labels --reference_files ../../../data_preparation/DATA/BIO_arg_timestamps/Debate27.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate28.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate29.txt
        fi
        echo "##########"

        done
    done

