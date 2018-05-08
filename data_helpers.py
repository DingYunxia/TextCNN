import numpy as np
import re
import itertools
from collections import Counter
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    #re.sub正则表达式
    #strip()指去掉首尾指定字符，无参数默认是去掉首尾的空格
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos",encoding='gb18030',errors='ignore').readlines())#将文件内容转换成列表形式存储
    #print(positive_examples) #将positive_examples打印出来发现每句话的结尾都有一个换行符\n
    positive_examples = [s.strip() for s in positive_examples]#去掉每句话末尾的换行符
    #print(positive_examples)#将上面的positive_examples打印出来发现换行符已经成功除去
    
    negative_examples = list(open("./data/rt-polarity.neg",encoding='gb18030',errors='ignore').readlines())
    negative_examples = [s.strip() for s in negative_examples]
   
    # Split by words
    x_text = positive_examples + negative_examples#将已经处理过的积极和消极句子合并为x_text
    x_text = [clean_str(sent) for sent in x_text]#调用clean_str()函数对x_text文本进行处理
    x_text = [s.split(" ") for s in x_text]#对x_text中的单词用空格进行分开，一个句子是一个list，每个list中的元素是一个str类型
                                             #所有的列表又放在一个列表里，即x_text数据类型是list
    # Generate labels 标签生成
    positive_labels = [[0, 1] for _ in positive_examples]#进行标签处理，将positive_examples中的所有内容转换为[0,1]标签，positive_examples是一个list，列表中的元素[0,1]也是一个list
                                                         #将所有的[0,1]又放在一个列表里，即positive_labels数据类型是list
    negative_labels = [[1, 0] for _ in negative_examples]#和positive_examples的处理一样
    y = np.concatenate([positive_labels, negative_labels], 0) #将多组列表连接为一个数组即y是一个数组，数组里面的元素类型是list    
    #print(y)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    将所有的句子平铺成同样的长度，长度是由最长的句子所决定的
    Returns padded sentences.
    最后返回平铺后句子的长度
    """
    sequence_length = max(len(x) for x in sentences) #找出最长句子的长度
    padded_sentences = [] #定义一个类型为list的变量padded_sentences
    
    #将句子处理为同样的长度
    for i in range(len(sentences)):
        sentence = sentences[i] #取第i个句子
        num_padding = sequence_length - len(sentence) #需要平铺的数量，即当前句子与最长句子的长度差是多少
        new_sentence = sentence + [padding_word] * num_padding #新句子的长度=当前句子的长度+补充的指定词*该词的数量
        padded_sentences.append(new_sentence) #在padded_sentences后面追加new_sentence
    return padded_sentences #结果返回一个已经处理过的句子

def build_vocab(sentences):
    """
    sentence1 = ['i', 'love', 'mom', 'mom', 'me', 'loves', 'me']
    建立一个词库(字典)
    Builds a vocabulary mapping from word to index based on the sentences.
    单词映射表，通过词的索引找到这个词
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary 建立一个字典
    word_counts = Counter(itertools.chain(*sentences)) #得到整个数据集上的词频统计
    #Counter内的内容是一个字典Counter({'mom': 2, 'me': 2, 'i': 1, 'love': 1, 'loves': 1})
    #print(word_counts)
    '''
    Counter(itertools.chain(*sentences))是指将多个迭代器作为参数，但只返回单个迭代器，他产生所有参数迭代器的内容
    就好像他们是来自一个单一的序列
    返回的结果是每个词出现的次数
    '''
    # Mapping from index to word 索引到单词的映射
    vocabulary_inv = [x[0] for x in word_counts.most_common()] # 统计word_counts中每个词出现的次数，即统计词频
    # 从字典中取出第一个元素即单词，高词频的词出现在前面，同时进行了去重['mom', 'me', 'i', 'love', 'loves']
    #print(vocabulary_inv)
    #most_common(n)返回一个top N列表，如果n没有被指定，则返回所有元素，当所有元素计数值相同时，按照字母顺序排序
    # Mapping from word to index
    #根据词频对单词建立索引，高词频在前面，低词频出现在后面
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #print(vocabulary)
    #结果返回[vocabulary, vocabulary_inv]
    return [vocabulary, vocabulary_inv]
    #c = [vocabulary, vocabulary_inv]
    #print(c)

# =============================================================================
#     import collections
#     sentence1 = ['i', 'love', 'mom', 'mom', 'me', 'loves', 'me', 'me']
#     # 统计每个单词出现的次数
#     word_counts1 = collections.Counter(sentence1)
#     print(word_counts1)
#     print(word_counts1.most_common())
#     
#     #取出字典的第一个元素key值，并进行了去重操作
#     vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
#     print(vocabulary_inv1)
#     #vocabulary_inv1 = list(sorted(vocabulary_inv1))#根据字母表的顺序进行排序
#     #print(vocabulary_inv1)
#     # 为已经排序过的单词建立索引,根据词频进行统计，单词出现的次数越多，排序越靠前
#     vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}
#     print(vocabulary1)
#     print([vocabulary_inv1, vocabulary1])
# =============================================================================
    
def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
# =============================================================================
#     x1 = np.array([[vocabulary1[word1] for word1 in sentence1] for sentences1 in sentences])
#     print(x1)
# =============================================================================
    
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    #print(x) 
# =============================================================================
#     # 一个句子sentence里面的词在vocabulary里对应的索引位置，
#     #例如the government hasdecided这几个词对应在vocabulary中的索引是0,563,6
#     ab =  [vocabulary[word] for word in sentence]
#     #将所有的句子都转成索引的形式进行存储,是一个二维列表
#     cd = [[vocabulary[word] for word in sentence] for sentence in sentences]
#     #将cd生成的二维列表转换成二维数组
#     ef = np.array(cd)
#     print(ab)
#     print(cd)
#     print(ef)
#    
# =============================================================================
    # 因为labels已经是一个0,1的二维列表了，所以可以直接转换成数组(array)
    y = np.array(labels)
    #print(y)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels() #句子和标签调用了load_data_and_labels()函数进行处理得到
    #此处的sentence和labels分别对应load_data_and_labels()中最后返回的x_text和y 
    #print(sentences,labels)
    sentences_padded = pad_sentences(sentences) #sentences_padded调用 pad_sentences()函数，其中函数参数为sentences
    vocabulary, vocabulary_inv = build_vocab(sentences_padded) #vocabulary, vocabulary_inv调用build_vocab（）函数，参数为sentences_padded
    #print(vocabulary, vocabulary_inv)
    x, y = build_input_data(sentences_padded, labels, vocabulary) #x,y通过调用build_input_data()函数得到，函数参数为：sentences_padded,labels和vocabulary
    return [x, y, vocabulary, vocabulary_inv]

    # load_data()函数返回的是x,y,vocabulary和vocabulary_inv

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    batch_size和num_epochs的调参
    """
    data = np.array(data) #将数据转换成数组
    #print(data)
    data_size = len(data) #求出数据的长度
    num_batches_per_epoch = int(len(data)/batch_size) + 1 #每一个epoch中batch_size的数量
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch  #在每一个epoch中将数据随机打乱，就像洗牌
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
