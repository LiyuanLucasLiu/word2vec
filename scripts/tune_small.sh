count=0
for resample in 10 15 20
do
	for IterNum in 100 150 200
	do
		for negative in 5 10
		do
			for dropout in 0.4 0.5 0.6
			do
			# for learning_rate in 0.025 0.0125 0.008
			# do
				echo $IterNum " " $negative " " $resample " " $dropout " " $count " in 54 "
				>&2 echo $count " in 54 "
			# echo "kl:"
				# echo "l"
				# ../bin/rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 16 -NONE_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				echo "a"
				../bin/armodify -train /shared/data/ll2/CoType/data/intermediate/KBP/pure_train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/pure_test.data -threads 16 -NONE_idx 6 -cleng 100 -lleng 300 -negative $negative -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 0 -debug 1 -instances 133955 -test_instances 2031 -dropout $dropout
				# echo "3"
				count=$((count + 1))
			# done
			done
		done
	done
done