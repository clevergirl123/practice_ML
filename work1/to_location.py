import os
import csv
import shutil

root_path = r'F:\Practice\neu-dataset'
train_file_path = r'F:\Practice\test'
test_file_path = r'F:\Practice\train'
in_path = r'F:\学习\大三\大三下\生产实习\work1\all_picture_random.csv'
test_path =  r'F:\学习\大三\大三下\生产实习\work1\test_picture_random_1.csv'
train_path =  r'F:\学习\大三\大三下\生产实习\work1\train_picture_random_1.csv'

def to_location(root_path, train_file_path, test_file_path, in_path, test_path, train_path):
	pictures = list(csv.reader(open(in_path, 'r')))
	number = len(pictures)
	limit = number * 0.3 / 20
	type_set = set()
	count = {}
	for record in pictures:
		temp_set = set()
		temp_set.add(record[1])
		p_name = os.path.basename(record[0])
		if temp_set.issubset(type_set):
			if count[record[1]] < limit:
				count[record[1]] += 1
				with open(test_path, 'a+', encoding='utf-8') as f:
					des_path = os.path.join(test_file_path, p_name)
					shutil.copy(record[0], des_path)
					f.write(des_path + ',' + record[1] + '\n')
			else:
				with open(train_path, 'a+', encoding='utf-8') as f:
					des_path = os.path.join(train_file_path, p_name)
					shutil.copy(record[0], des_path)
					f.write(des_path + ',' + record[1] + '\n')
		else:
			type_set.add(record[1])
			count[record[1]] = 1
			with open(test_path, 'a+', encoding='utf-8') as f:
				des_path = os.path.join(test_file_path, p_name)
				shutil.copy(record[0], des_path)
				f.write(des_path + ',' + record[1] + '\n')
		

def mkdirs(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return True

if __name__ == '__main__':
	with open(test_path, 'w+') as f:
		f.write('')
	with open(train_path, 'w+') as f:
		f.write('')
	if mkdirs(train_file_path) and mkdirs(test_file_path):
		to_location(root_path, train_file_path, test_file_path, in_path, test_path, train_path)
