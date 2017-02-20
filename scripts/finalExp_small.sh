# count=0
# for IterNum in 20 30 40 50
# do
# 	for resample in 15
# 	do
# 		for negative in 10
# 		do
# 			for dropout in 0.2 0.25 0.3 0.35 0.4
# 			do
# 			# for learning_rate in 0.025 0.0125 0.008
# 			# do
# 				# echo $IterNum " " $negative " " $resample " " $dropout " " $count " in 54 "
# 				>&2 echo $count " in 16 "
# 			# echo "kl:"
# 				# echo "l"
# 				# ../bin/rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 16 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
# 				# echo "a"
# 				# ../bin/armodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 30 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
# 				../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 16 -NONE_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout
# 				count=$((count + 1))
# 			# done
# 			done
# 		done
# 	done
# done
# echo 'small'
count=0
for IterNum in 30 50 70 100 130
do
	for resample in 15
	do
		for negative in 10 5 20
		do
			for dropout in 0.13 0.15 0.17
			do
			# for learning_rate in 0.025 0.0125 0.008
			# do
				# echo $IterNum " " $negative " " $resample " " $dropout " " $count " in 54 "
				>&2 echo $count " in 20 "
			# echo "kl:"
				# echo "l"
				# ../bin/rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 16 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				# echo "a"
				# ../bin/armodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 30 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 40 -NONE_idx 1 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 106684 -test_instances 3423 -val_instances 380
				# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_pattern_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_pattern_eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/pure_pattern_val.data -threads 40 -NONE_idx 1 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 95210 -test_instances 279 -val_instances 31
				# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 30 -NONE_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211
				# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/eva.data -val /shared/data/ll2/CoType/data/intermediate/KBP/val.data -threads 30 -NONE_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 1 -debug 1 -dropout $dropout -instances 225977 -test_instances 1900 -val_instances 211
				../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 40 -none_idx 1 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 106684 -test_instances 3423 -val_instances 380 -error_log 1
				# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/BioInfer/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 30 -none_idx 141 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout -instances 94304 -test_instances 3408 -val_instances 378
				count=$((count + 1))
			# done
			done
		done
	done
done

# ../bin/finalExp -train /shared/data/ll2/CoType/data/intermediate/NYT/train.data -test /shared/data/ll2/CoType/data/intermediate/NYT/eva.data -val /shared/data/ll2/CoType/data/intermediate/NYT/val.data -threads 40 -none_idx 1 -cleng 200 -lleng 500 -negative 10 -resample 15 -ignore_none 0 -iter 100 -alpha 0.025 -normL 0 -debug 1 -dropout 0.6 -instances 106684 -test_instances 3423 -val_instances 380
