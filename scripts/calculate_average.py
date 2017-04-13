_author__ = 'S2free'

import argparse
import json


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', nargs='+', default=['results/new_wTune_KBP_large_rep_4.log', 'results/new_wTune_KBP_large_rep_5.log'])
	args = parser.parse_args()

	for file_name in args.input_file:
		with open(file_name, 'r') as f:
			lines = map(lambda t: map(lambda x: float(x), t.split(',')), f.readlines())
		sstr = ''
		for ind in range(0, len(lines[0])):
			sstr = sstr + str(sum(map(lambda t: t[ind], lines)) / len(lines)) + ', '
		sstr = sstr + '\n'
		print(sstr)
		