import os
import csv
import random
import shutil

# 原始数据，（图片原绝对路径，标签）
file_name = r'F:\学习\大三\大三下\生产实习\work1\my_picture.csv'
# 图片文件目录
root_path = r'C:\Users\LQB\Desktop\new_data'
# 和file_name是同一个路径，啊
in_path = r'F:\学习\大三\大三下\生产实习\work1\my_picture.csv'
# 训练图片目标文件夹
train_file_path = r'F:\Practice\my_data\train_new'
# 测试图片目标文件夹
test_file_path = r'F:\Practice\my_data\test_new'
# 测试图片路径、标签 表
test_path =  r'F:\Practice\my_data\test.csv'
# 训练图片路径、标签 表
train_path =  r'F:\Practice\my_data\train.csv'

label_map1 = {'bear':0,
			'bicycle':1,
			'bird':2,
			'car':3,
			'cow':4,
 			'elk':5,
			'fox':6,
			'giraffe':7,
			'horse':8,
			'koala':9,
			'lion':10,
			'monkey':11,
			'plane':12,
			'puppy':13,
			'sheep':14,
			'statue':15,
			'tiger':16,
			'tower':17,
			'train':18,
			'whale':19,
			'zebra':20}
label_map = {'bear':0,
			'bird':1,
			'car':2,
			'cow':3,
 			'elk':4,
			'fox':5,
			'giraffe':6,
			'horse':7,
			'koala':8,
			'lion':9,
			'monkey':10,
			'plane':11,
			'puppy':12,
			'sheep':13,
			'statue':14,
			'tiger':15,
			'tower':16,
			'train':17,
			'whale':18,
			'zebra':19,
			'bicycle':20}

def tran(root_path, file_name, father = 'ds2018'):
	files = os.listdir(root_path)
	for file in files:
		path = os.path.join(root_path,file)
		if os.path.isdir(path):
			tran(path, file_name, file)
		else:
			with open(file_name, 'a+',encoding = 'utf-8') as f:
				f.write(path + ',' + father + '\n')

def to_random_sample(in_path):
	pictures = list(csv.reader(open(in_path, 'r')))
	new_pictures = random.sample(pictures, len(pictures))
	return new_pictures

def to_location(root_path, train_file_path, test_file_path, test_path, train_path, pictures):
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
					f.write(des_path + ',' + str(label_map[record[1]]) + '\n')
			else:
				with open(train_path, 'a+', encoding='utf-8') as f:
					des_path = os.path.join(train_file_path, p_name)
					shutil.copy(record[0], des_path)
					f.write(des_path + ',' + str(label_map[record[1]]) + '\n')
		else:
			type_set.add(record[1])
			count[record[1]] = 1
			with open(test_path, 'a+', encoding='utf-8') as f:
				des_path = os.path.join(test_file_path, p_name)
				shutil.copy(record[0], des_path)
				f.write(des_path + ',' + str(label_map[record[1]]) + '\n')
		

def mkdirs(path):
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		print(path + ' 创建成功')
		return True
	else:
		print(path + ' 目录已存在，请移除已存在目录')
		return False
	
if __name__ == '__main__':
	with open(file_name, 'w+') as f:
		f.write('')
	tran(root_path, file_name)
	with open(test_path, 'w+') as f:
		f.write('')
	with open(train_path, 'w+') as f:
		f.write('')
	if mkdirs(train_file_path) and mkdirs(test_file_path):
		to_location(root_path, train_file_path, test_file_path, test_path, train_path, to_random_sample(in_path))
