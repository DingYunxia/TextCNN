from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len #数量*最大长度
    vocabulary_inv  # dict {str:int} 数据类型为字典，key是词，value是这个词对应的索引
    num_features    # Word vector dimensionality 词向量的维度 300维                 
    min_word_count  # Minimum word count                        
    context         # Context window size 文本窗口的大小
    """
    # 根目录是word2vec_models(即文件夹的名字就是这个)的形式命名xxminwords_xxcontext
    model_dir = 'word2vec_models'
    # 模型的名字以"xxfeatures_"
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    #print(model_name)
    #将模型名字的路径拼接起来
    model_name = join(model_dir, model_name)
    #print(model_name)
    #一个判断语句，如果这个模型已经存在做什么操作，不存在又怎么办
    if exists(model_name):
        #该模型存在,embedding_model就是直接调用Word2Vec中的load()函数
        embedding_model = word2vec.Word2Vec.load(model_name)
        #print(embedding_model)输出结果为：Word2Vec(vocab=18765, size=300, alpha=0.025)
        #得到的结果是词汇表中有18765个词，词向量的维度是300，学习率α是0.025
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
        #打印出的结果为：Loading existing Word2Vec model '300features_1minwords_10context'
    else:
        # Set values for various parameters 为各个参数设置值
        num_workers = 2       # Number of threads to run in parallel
        downsampling = 1e-3   # Downsample setting for frequent words
                              #常用词的下采样设置 downsampling=0.001
                              
        # Initialize and train the model 初始化并训练模型
        print("Training Word2Vec model...")
        #
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                            size=num_features, min_count = min_word_count, \
                            window = context, sample = downsampling)
        
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
        
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)
    
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in vocabulary_inv])]
    return embedding_weights

if __name__=='__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)

