_author__ = 'S2free'

import argparse
import json


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', nargs='+', default=['results/new_eTune_KBP_large.log', 'results/new_wTune_KBP_large.log', 'results/new_eTune_NYT_large.log', 'results/new_wTune_NYT_large.log'])
	args = parser.parse_args()

	for file_name in args.input_file:
		with open(file_name, 'r') as f:
			lines = f.readlines()
		with open(file_name, 'w') as f:
			for ind in range(0, len(lines)):
				if ind%5 != 0:
					f.write(lines[ind])