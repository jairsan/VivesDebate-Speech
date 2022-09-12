train_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate1.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate2.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate3.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate4.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate5.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate6.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate7.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate8.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate9.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate10.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate11.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate12.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate13.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate14.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate15.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate16.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate17.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate18.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate19.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate20.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate21.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate22.txt"
dev_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt"

#cant be loaded using fp32, even for BS=1
#python3 ../../../src/train/train_transformers_classifier.py  facebook/xlm-roberta-xl "$train_files" "$dev_files" XLM-ROBERTA-XL

#python3 ../../../src/train/train_transformers_classifier.py  --model_name xlm-roberta-large --train_files "$train_files" --eval_files "$dev_files" --output_dir_name XLM-ROBERTA-LARGE

#With punc datasets
#python3 ../../../src/train/train_transformers_classifier.py  PlanTL-GOB-ES/roberta-base-ca "$train_files" "$dev_files" BERTa


#for num_spans in 5 10;
for num_spans in 5;
do
  #With spans datasets
  spans_train=../SHAS-multi/spans/train.maxlen$num_spans/
  spans_dev=../SHAS-multi/spans/dev.maxlen$num_spans/

  output_dir=BERTa_spans$num_spans
  rm -r "$output_dir"_models "$output_dir"_tokenizer

  python3 ../../../src/train/train_transformers_classifier.py  --model_name PlanTL-GOB-ES/roberta-base-ca \
   --train_files "$train_files" --eval_files "$dev_files" --output_dir_name $output_dir \
   --generate_train_datasets_from_spans_folder $spans_train \
   --generate_eval_datasets_from_spans_folder $spans_dev \
   --learning_rate 5e-5 \
   --per_device_train_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --per_device_eval_batch_size 32 \
   --num_train_epochs 10

 done
