count=0
for IterNum in 200 250 300
do
	for resample in 15
	do
		for negative in 10
		do
			for dropout in 0.6 0.65 0.7
			do
			# for learning_rate in 0.025 0.0125 0.008
			# do
				echo $IterNum " " $negative " " $resample " " $dropout " " $count " in 54 "
				>&2 echo $count " in 54 "
			# echo "kl:"
				# echo "l"
				# ../bin/rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 16 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				echo "a"
				../bin/armodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 30 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				../bin/armodify -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/test.data -threads 30 -NONE_idx 6 -cleng 200 -lleng 500 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -dropout $dropout
				count=$((count + 1))
			# done
			done
		done
	done
done