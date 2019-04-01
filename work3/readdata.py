import paddle.v2 as paddle
from multiprocessing import cpu_count
import csv

class MyReader:
    def __init__(self,imageSize):
        self.imageSize = imageSize

    def train_mapper(self,sample):
        img,label = sample

        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img,256,self.imageSize,True)
        return img.flatten().astype('float32'),label

    def test_mapper(self,sample):

        img,label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img,256,imageSize,False)
        return img.flatten().astype('float32'),label

    def train_reader(self,train_list,buffered_size=1024):
        def reader():
            with open(train_list,'r') as f:
                read_train = csv.reader(f)
                for item in read_train:
                    img_path,lab = item
                    yield img_path,int(lab)
        return paddle.reader.xmap_readers(self.train_mapper,reader,cpu_count(),buffered_size)
    def test_reader(self,test_list,buffered_size=1024):
        def reader():
            with open(test_list,'r') as f:
                read_test = csv.reader(f)
                for item in read_test:
                    img_path,lab = item
                    yield img_path,lab
        return paddle.reader.xmap_readers(self.test_mapper,reader,cpu_count(),buffered_size)


if __name__ == '__main__':
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 32
    # 总的分类名称
    all_class_name = 'vegetables'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize
    paddleUtil = PaddleUtil()

    # *******************************开始训练**************************************
    myReader = MyReader(imageSize=imageSize)
    # # parameters_path设置为None就使用普通生成参数,
    trainer = paddleUtil.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    trainer_reader = myReader.train_reader(train_list="../data/%s/trainer.list" % all_class_name)
    test_reader = myReader.test_reader(test_list="../data/%s/test.list" % all_class_name)

    paddleUtil.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,
                             trainer_reader=trainer_reader, test_reader=test_reader)