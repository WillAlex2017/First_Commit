preprocess.py
    目的：数据预处理
    用法：preprocess.py -f path/to/dataset/ [option]
        Option                     | Desciption
        -------------             | ----------
        `--columns`             | 数据列的顺序，'u'代表user，'i'代表项目，'r'代表评分，'t'代表时间戳。例如，uirt代表列的顺序是 用户、项目、评分、时间戳
        `--sep`                     | 列的风格符
        `--min_user_activity` |  用户最少的交互项目数，小于则剔除. 默认: 2
        `--min_item_pop`    |    项目最少被交互的用户数，小于则剔除. 默认: 5
        `--val_size`             |  交叉验证集的用户数占所有用户的比率，取值(0,1)。默认：0.1
        `--test_size`            | 测试集的用户数占所有用户的比率，取值(0,1)。默认：0.1
        `--seed`                 | 随机种子数，用于随机划分数据集
        
        
train.py
    目的：训练模型（马尔科夫、基于用户不需要训练）
    用法：train.py -d path/to/preprocessed_dataset  -m MODEL_NAME [model_option]
    
        可选模型有：
        RNN
            （-m RNN）
            `--r_t [LSTM, GRU, Vanilla]`                                | 循环神经网络的类型 (default is GRU)
            `--r_l size_of_layer1-size_of_layer2-etc.`              | 网络隐层参数维度
            `--u_m [rmsprop, nesterov, adam]`                    | 优化算法。默认：adam
            `--u_l float`                                                       | 学习率。默认：0.001
            `--u_rho float`                                                  | rho参数。默认：0.9
            `--u_b1 float`                                                    | Beta 1 参数，用于Adam。默认：0.9
            `--u_b2 float`                                                    | Beta 2 参数，用于Adam。默认：0.999
            `--n_dropout P`                                                 | Dropout 屏蔽概率。默认：0
        矩阵分解MF
            （-m MF）
            `-H int`             | 矩阵分解的维度。默认：20
            `-l val`              | 学习率。默认：0.01
        马尔科夫链
            （-m MM）
        基于用户
            （-m UKNN）
            `ns int`                |   近邻数量。默认：20
            
            
test.py
    目的：用测试集测试训练出来的样本
    用法：test.py -d path/to/preprocessed_dataset -m MODEL_NAME
        可加上-k参数，指定模型的推荐列表长度，即指定TopN的大小。
