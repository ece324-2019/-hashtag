from gensim.models.keyedvectors import KeyedVectors
# from pathlib import Path
# from gensim.test.utils import datapath, get_tmpfile
# from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('model_test.txt')
# tmp_file = get_tmpfile("tim.txt")
# _ = glove2word2vec(glove_file, tmp_file)


# model = KeyedVectors.load_word2vec_format('model_test.txt', binary=False)
# model = KeyedVectors.load_word2vec_format('./model_test.txt', binary=False)

print("blogto")
print(model.most_similar(positive='blogto'))
# print('----------------')
# print('中国')
# print(model.most_similar(positive='中国'))
