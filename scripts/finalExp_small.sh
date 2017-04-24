count=0
for IterNum in 20 30 50 70 #80 90 100 
do
	for resample in 15
	do
		for negative in 10
		do
			for dropout in 0.2 0.4 0.6
			do
				>&2 echo $count " in 12 "

				 # ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 30 -none_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1 >> results/wTune_KBP_large.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 20 -none_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1 >> results/eTune_KBP_large.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -val /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -test /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 20 -none_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -val_instances 1900 -test_instances 211 -special_none 0 -error_log 1 >> results/eTune_KBP_large.log

				# >&2 echo $count " NYT "

				../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 20 -none_idx 0 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1 >> results/n_wTune_NYT_large.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 20 -none_idx 0 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1 >> results/n_eTune_NYT_large.log
				

				../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/BioInfer/train.data -test /shared/data/ll2/CoType/data/intermediate/BioInfer/eva.data -val /shared/data/ll2/CoType/data/intermediate/BioInfer/val.data -threads 20 -none_idx 200 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 2 -dropout $dropout -instances 94293 -test_instances 765 -val_instances 84 -special_none 0 -error_log 1 >> results/eTune_Bio_large.log

				count=$((count + 1))

			done
		done
	done
done
../bin/ReHession -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 30 -none_idx 6 -cleng 150 -lleng 250 -negative 9 -resample 20 -ignore_none 0 -iter 15 -alpha 0.025 -normL 0 -debug 1 -dropout 0.4 -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1

# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/BioInfer/train.data -test /shared/data/ll2/CoType/data/intermediate/BioInfer/eva.data -val /shared/data/ll2/CoType/data/intermediate/BioInfer/val.data -threads 35 -none_idx 102 -cleng 200 -lleng 500 -negative 10 -resample 15 -ignore_none 1 -iter 20 -alpha 0.025 -normL 0 -debug 2 -dropout 0.2 -instances 94304 -test_instances 3408 -val_instances 378 -special_none 0

# ../bin/tdExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -output_file ../data/KBP_td.json -threads 30 -none_idx 6 -cleng 150 -lleng 250 -negative 7 -resample 20 -ignore_none 0 -iter 14 -alpha 0.025 -normL 0 -debug 2 -dropout 0.3 -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0

# ../bin/tdSave -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -output_file ../data/KBP_td.model -threads 30 -none_idx 6 -cleng 150 -lleng 250 -negative 7 -resample 20 -ignore_none 0 -iter 14 -alpha 0.025 -normL 0 -debug 2 -dropout 0.3 -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -binary 1

# ../bin/tdNew -input_model ../data/KBP_td.model -output_file ../data/KBP_sample_td.json -val /shared/data/ll2/CoType/data/intermediate/KBP/sample.data -threads 30 -none_idx 6 -cleng 150 -lleng 250 -negative 7 -resample 20 -ignore_none 0 -iter 14 -alpha 0.025 -normL 0 -debug 2 -dropout 0.3 -instances 225977 -test_instances 1900 -val_instances 14 -special_none 0 -binary 1


# ../bin/recs -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -output_file ../data/NYT_re.json -threads 35 -none_idx 0 -cleng 150 -lleng 250 -negative 2 -resample 15 -ignore_none 0 -iter 20 -alpha 0.025 -normL 0 -debug 2 -dropout 0.3 -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0

# ../bin/tdExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -output_file ../data/NYT_td.json -threads 20 -none_idx 0 -cleng 150 -lleng 250 -negative 2 -resample 15 -ignore_none 0 -iter 20 -alpha 0.025 -normL 0 -debug 1 -dropout 0.3 -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1