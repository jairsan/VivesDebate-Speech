train_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate1.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate2.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate3.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate4.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate5.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate6.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate7.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate8.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate9.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate10.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate11.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate12.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate13.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate14.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate15.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate16.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate17.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate18.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate19.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate20.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate21.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate22.txt"
dev_files="../../../data_preparation/DATA/BIO_arg_timestamps/Debate24.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate25.txt ../../../data_preparation/DATA/BIO_arg_timestamps/Debate26.txt"

python3 ../../../src/train/train_transformers_classifier.py  PlanTL-GOB-ES/roberta-base-ca "$train_files" "$dev_files" tmp_transformers
