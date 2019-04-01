import os

root_path = 'F:\\Practice\\neu-dataset'
file_name = 'all_picture.csv'

def tran(root_path, file_name, father = 'neu-dataset'):
	files = os.listdir(root_path)
	for file in files:
		path = os.path.join(root_path,file)
		if os.path.isdir(path):
			tran(path, file_name, file)
		else:
			with open(file_name, 'a+',encoding = 'utf-8') as f:
				f.write(path + ',' + father + '\n')
			
if __name__ == '__main__':
	with open(file_name, 'w+') as f:
		f.write('')
	tran(root_path, file_name)
