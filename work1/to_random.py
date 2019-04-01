import random
import csv

in_path = r'F:\学习\大三\大三下\生产实习\work1\all_picture.csv'
out_path = r'F:\学习\大三\大三下\生产实习\work1\all_picture_random.csv'

def to_random(in_path, out_path):
	pictures = list(csv.reader(open(in_path, 'r')))
	random.shuffle(pictures)
	for record in pictures:
		with open(out_path,'a+',encoding='utf-8') as f:
			f.write(record[0] + ',' + record[1] + '\n')

def to_random_sample(in_path, out_path):
	pictures = list(csv.reader(open(in_path, 'r')))
	new_pictures = random.sample(pictures, len(pictures))
	for record in new_pictures:
		with open(out_path,'a+',encoding='utf-8') as f:
			f.write(record[0] + ',' + record[1] + '\n')
			
if __name__ == '__main__':
	with open(out_path, 'w+') as f:
		f.write('')
	to_random_sample(in_path,out_path)
