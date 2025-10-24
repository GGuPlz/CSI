import warnings


class DefaultConfig(object):    
    #system_path = '/media/public/z509/g24/gyt/CSI'
    system_path = 'C:/Users/gyt/OneDrive/Desktop/code/CSI'
    
    ##path for gyt pc windows
    data = system_path + '/CSI_data'
    train_data = system_path + '/CSI_data/train'
    test_data = system_path + '/CSI_data/test'

    #load_model_path='/media/public/z509/g23/wyt/YOLOV11/2WIFI-2D/run/20250426/main_small/结果/best.pth'
    #load_model_path='/media/public/z509/g24/gyt/CSI/run/0928/best.pth'
    load_model_path = None
    
    batch_size = 32#指定每个批次的样本数量，即一次性送入模型进行训练或推理的样本数量
    use_gpu = True    #指定是否使用 GPU 进行计算
    num_workers = 0  #指定数据加载过程中使用的线程数，即同时加载数据的线程数量
    use_MSE = False
    print_freq = 20   #指定训练过程中每隔多少个批次打印一次训练信息

    debug_file = 'tmp/debug'    #没用到
    save_path = system_path + '/run/1004'  #模型保存路径
    result_file = system_path + '/run/1004/results.csv' 
    resultvideo_file = system_path + '/run/1004'

    max_epoch = 250   #训练的最大轮数
    # lr = 0.00002      #学习率，即每次更新模型参数的步长大小
    lr = 0.00002 
    # lr = 0.001
    lr_decay = 0.95   #学习率衰减因子，用于控制学习率的衰减速度
    weight_decay = 1e-6 #权重衰减参数，用于控制模型的正则化程度
 
'''
这段代码定义了一个parse方法,用于解析传入的参数kwargs并将其设置为类的属性。具体步骤如下:
遍历参数字典kwargs中的每个键值对。
对于每个键值对，检查是否存在对应的属性，如果不存在则发出警告。
使用setattr方法将参数设置为类的属性。
打印出用户配置的属性和对应的值，以便确认参数是否正确设置。
这个方法通常用于设置类的参数，可以根据需要在类的实例化过程中使用。
'''
def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
