import os
import csv

in_path = r'F:\学习\大三\大三下\生产实习\work1\all_picture_random.csv'
test_path =  r'F:\学习\大三\大三下\生产实习\work1\test_picture_random.csv'
train_path =  r'F:\学习\大三\大三下\生产实习\work1\train_picture_random.csv'
			
def to_classify(in_path, test_path, train_path):
	pictures = list(csv.reader(open(in_path, 'r')))
	number = len(pictures)
	limit = number * 0.3 / 20
	print('limit:' + str(limit))
	type_set = set()
	count = {}
	for record in pictures:
		temp_set = set()
		temp_set.add(record[1])
		print('temp_set:' + str(temp_set))
		if temp_set.issubset(type_set):
			if count[record[1]] < limit:
				count[record[1]] += 1
				with open(test_path, 'a+', encoding='utf-8') as f:
					f.write(record[0] + ',' + record[1] + '\n')
			else:
				with open(train_path, 'a+', encoding='utf-8') as f:
					f.write(record[0] + ',' + record[1] + '\n')
		else:
			type_set.add(record[1])
			count[record[1]] = 1
			with open(test_path, 'a+', encoding='utf-8') as f:
				f.write(record[0] + ',' + record[1] + '\n')

if __name__ == '__main__':
	with open(test_path, 'w+') as f:
		f.write('')
	with open(train_path, 'w+') as f:
		f.write('')
	to_classify(in_path, test_path, train_path)
