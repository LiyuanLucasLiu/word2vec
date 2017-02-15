count=0
for IterNum in 120 140
do
	for lambda2 in 1 0.3 3
	do
		for resample in 5 10 15 20 25
		do
			# for learning_rate in 0.025 0.0125 0.008
			# do
			echo $IterNum " " $lambda2 " " $resample " " $count " in 40 "
			>&2 echo $count " in 40 "
			# echo "kl:"
			../bin/rmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/test.data -threads 20 -NONE_idx 6 -cleng 30 -lleng 50 -lambda2 $lambda2 -resample $resample -ignore_none 0 -iter $IterNum -alpha 0.025 -normL 2 -debug 1
			# echo "margin:"
			# ../bin/mmodify -train /shared/data/ll2/CoType/data/intermediate/KBP/train.data -test /shared/data/ll2/CoType/data/intermediate/KBP/test.data -threads 20 -NONE_idx 6 -cleng 30 -lleng 50 -lambda1 $lambda1 -resample $resample -ignore_none 1 -iter $IterNum -alpha 0.025 -normL 2 -debug 1
			count=$((count + 1))
			# done
		done
	done
done 