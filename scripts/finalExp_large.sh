count=0
for IterNum in 10 20 40 70 #20 30 50 70 
do
	for resample in 15
	do
		for negative in 10
		do
			for dropout in 0.2 0.4 0.6
			do
				>&2 echo $count " in 12 "

				 # ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 30 -none_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1 >> results/wTune_KBP_large.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 20 -none_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1 >> results/eTune_KBP_large_s.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 20 -none_idx 6 -cleng 50 -lleng 100 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211 -special_none 0 -error_log 1 >> results/eTune_KBP_large_t.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -val /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -test /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 20 -none_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -val_instances 1900 -test_instances 211 -special_none 0 -error_log 1 >> results/eTune_KBP_large.log

				# >&2 echo $count " NYT "

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 30 -none_idx 0 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1 >> results/wTune_NYT_large.log

				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 20 -none_idx 0 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1 >> results/eTune_NYT_large_s.log
				
				# ../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 20 -none_idx 0 -cleng 50 -lleng 100 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 530767 -test_instances 3423 -val_instances 380 -special_none 0 -error_log 1 >> results/eTune_NYT_large_t.log
				
				../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/BioInfer/train.data -test /shared/data/ll2/CoType/data/intermediate/BioInfer/eva.data -val /shared/data/ll2/CoType/data/intermediate/BioInfer/val.data -threads 20 -none_idx 200 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 94293 -test_instances 765 -val_instances 84 -special_none 0 -error_log 1 >> results/eTune_Bio_large_s.log

				../bin/wfinalExp -train /shared/data/ll2/CoType/data/intermediate/BioInfer/train.data -test /shared/data/ll2/CoType/data/intermediate/BioInfer/eva.data -val /shared/data/ll2/CoType/data/intermediate/BioInfer/val.data -threads 20 -none_idx 200 -cleng 50 -lleng 100 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 94293 -test_instances 765 -val_instances 84 -special_none 0 -error_log 1 >> results/eTune_Bio_large_t.log
				count=$((count + 1))

			done
		done
	done
done