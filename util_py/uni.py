import json 
import os

c = 0
wfile = open('../data/ast_nl.json', 'w')
for rt, dirs, files in os.walk(r'../out/'):
	for f in files:
		file = open(os.path.join(rt, f) , 'r')
		data = file.read()
		lines = data.split('\n')
		for line in lines:
			try:
				job = json.loads(line)
			except Exception as e:
				continue
			c += 1
			if c % 10000 == 0:
				print c
			newjob = {}
			newjob['root'] = job['root']
			newjob['nl'] = job['comment']
			newline = json.dumps(newjob) + '\n'
			wfile.write(newline)
print c