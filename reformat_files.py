import os
import io
import re
import string

pat = re.compile(r"([{}])".format(re.escape(string.punctuation)))
space_ = re.compile(r"\s+")
files = os.listdir('data')

for file in files:
	if file != ".DS_Store":
		l = []
		print(file)
		with io.open(f"data/{file}", "r", encoding="utf-8") as f:
			for index,line in enumerate(f.readlines()):
				line = line.replace("\n","")
				line = pat.sub(r" \1 ", line)
				line = space_.sub(" ", line)
				line = line.strip()
				l.append(line)
		with io.open(f"d/{file}", "w", encoding="utf-8") as f:
			for line in l:
				try:
					f.write(line+"\n")
				except Exception as e:
					print(line)