
# coding: utf-8

import os
import numpy as np
import collections
import tensorflow as tf

class CHARRNN:
    '''Char-rnn generate chinese ancient poemtry
    Args:
       rnn_size:number of hidden units in cell
       vocal_size:vocabulary size
       lr:learning rate
       n_chunk:how many batches is included in training file
       num_to_word:num-to-word dictionary used at test phase
       word_to_num:word-to-num dictionary used at test phase
       layer_num:layers of RNN
    '''
    def __init__(self,**param):
        self.rnn_size=param.get('rnn_size',128)
        self.num_layers=param.get('num_layers',2)
        self.lr=param.get('learning rate',0.002)
        self.batch_size=param.get('batch_size',64)
        self.vocab_size=param.get('vocab_size',None)
        self.n_chunk=param.get('n_chunk',None)
        self.word_to_num=param.get('word_to_num',None)
        self.num_to_word=param.get('num_to_word',None)
        self.layer_num=param.get('layer_num',1)
        self.cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size,reuse=tf.get_variable_scope().reuse)
        if self.layer_num>1:
            cells=[self.cell for _ in range(self.layer_num)]
            self.cell=tf.contrib.rnn.MultiRNNCell(cells)
     
    def _weights(self,shape,name='weights'):
        initializer=tf.random_normal_initializer(mean=0.,stddev=0.5)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)
    
    def _bias(self,shape,name='bias'):
        initializer=tf.constant_initializer(0.)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)

    def output(self,state,reuse):
        '''Build Model
        W:output layer weights
        b:output layer bias
        model structure: 1 embedding layer + RNN layer(s) + 1 output layer
        reuse variable is used to reuse the scope variable
        '''
        with tf.variable_scope('output',reuse=reuse):
            self.W=self._weights([self.rnn_size,self.vocab_size])
            self.b=self._bias([self.vocab_size])
            inputs=tf.contrib.layers.embed_sequence(self.train,self.vocab_size,self.rnn_size)
            outputs,last_state=tf.nn.dynamic_rnn(self.cell,inputs,initial_state=state)
            output=tf.reshape(outputs,[-1,self.rnn_size])
            logits=tf.matmul(output,self.W)+self.b
        return (logits,last_state)
          
    def train(self,x,y,reuse):
        '''Train model
        cross entropy is used as loss function
        clip gradients by value: 5. and adjust learning rate along with training steps increase
        After training store variables to 'poem_charrnn.ckpt'
        '''
        initial_state=self.cell.zero_state(self.batch_size,tf.float32)  
        self.train=tf.placeholder(tf.int32,[self.batch_size,None])
        self.target=tf.placeholder(tf.int32,[self.batch_size,None])    
        with tf.name_scope('train'):
            targets=tf.reshape(self.target,[-1])
            logits,_,=self.output(initial_state,reuse)
            loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[targets],[tf.ones_like(targets, dtype=tf.float32)])
            cost=tf.reduce_mean(loss)
            learning_rate = tf.Variable(self.lr,trainable=False)
            tvars =tf.trainable_variables()
            grads, _ =tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
            optimizer =tf.train.AdamOptimizer(self.lr)
            train_op=optimizer.apply_gradients(zip(grads, tvars))
            saver=tf.train.Saver(tf.all_variables())
            init=tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                error_=[]
                for epoch in range(60):
                    sess.run(tf.assign(learning_rate, self.lr * (0.97 ** epoch)))
                    for i in range(self.n_chunk):
                        train_loss,_=sess.run([cost,train_op],feed_dict={self.train:x[i],self.target:y[i]})
                        if i%600==0:
                            print('epoch: {}, Step: {}, loss:{:4}'.format(epoch,i,train_loss))
                        if i%50==0:
                            error_.append(train_loss)
                saver.save(sess,os.path.join(os.getcwd(), 'poem_charrnn.ckpt'))
            return error_
                            
    def to_word(self,weights):
        t=np.cumsum(weights)
        s=np.sum(weights)
        sample=int(np.searchsorted(t,np.random.rand(1)*s))
        return self.num_to_word[sample]
    
    def predict(self,reuse):
        '''Generate Poetry
        Args:
        reuse - whether reuse variables within model 
        use trained model to generate poetry - load parameters from ckpt file
        initialize with charater:'[' and predict the next charater by taking the hidden state from previous step as input
        '''
        self.batch_size=1
        self.train=tf.placeholder(tf.int32,[self.batch_size,None])        
        with tf.name_scope('predict'):
            initial_state=self.cell.zero_state(self.batch_size,tf.float32)  
            gx=np.array([[self.word_to_num.get('[')]])
            logits,last_state=self.output(initial_state,reuse)
            probs=tf.nn.softmax(logits)
            init=tf.global_variables_initializer()
            sess=tf.Session()
            sess.run(init)
            saver=tf.train.Saver()
            saver.restore(sess,os.path.join(os.getcwd(), 'poem_charrnn.ckpt')) 
            [probs_,state_]=sess.run([probs,last_state],feed_dict={self.train:gx})
            word=self.to_word(probs_)
            poem=''
            while word !=']':
                poem+=word
                gx=np.zeros((1,1))
                gx[0,0]=self.word_to_num[word]
                [probs_,state_]=sess.run([probs,last_state],feed_dict={self.train:gx,initial_state:state_})
                word=self.to_word(probs_)
        return poem


def load_file(file_path):
    '''load file
    load file
       remove poetry title, unexpected symbols
    Return:
       a list with all desired poetrys contained
    '''
    poetry_file=file_path
    poetrys=[]
    with open(poetry_file,'r',encoding='utf-8',) as f:
        for line in f:
            try:
                title,content=line.strip().split(':')
                content=content.replace(' ','')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content: 
                    continue
                if len(content)<5 or len(content)>79:
                    continue
                content='['+content+']'
                poetrys.append(content)
            except Exception as e:
                pass

    poetrys=sorted(poetrys,key=lambda line: len(line))
    print('诗词总数:',len(poetrys))
    return poetrys

def prepare_data(poetrys,batch_size):
    '''divide data into batches
    Return:
       training data
       target data
       vocalbulary size
       dictionary:word_to_num
       dictionary:num_to_word
       number of batches
    '''
    all_words=[]
    for poetry in poetrys:
        all_words+=[word for word in poetry]
    counter=collections.Counter(all_words)
    count_pairs=sorted(counter.items(),key=lambda x:-x[-1])
    words,_=zip(*count_pairs)
    words=words[:len(words)]+(' ',)
    vocal_size=len(words)
    word_to_num=dict(zip(words,range(len(words))))
    num_to_word=dict(zip(range(len(words)),words))
    to_num=lambda word:word_to_num.get(word,len(words))
    poetrys_vector=[list(map(to_num,poetry)) for poetry in poetrys]
    n_chunk=len(poetrys_vector)//batch_size
    x_batches=[]
    y_batches=[]
    for i in range(n_chunk):
        start_index=i*batch_size
        end_index=start_index+batch_size
        batches=poetrys_vector[start_index:end_index]
        length=max(map(len,batches))
        xdata=np.full((batch_size,length),word_to_num[' '],np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])]=batches[row]
        ydata=np.copy(xdata)
        ydata[:,:-1]=xdata[:,1:]
        x_batches.append(xdata)
        y_batches.append(ydata)
    return x_batches,y_batches,vocal_size,n_chunk,word_to_num,num_to_word

def run():
    print('***loading data***')
    poetrys=load_file('poetry.txt')
    print('***data loaded***')
    x_batches,y_batches,vocal_size,n_chunk,word_to_num,num_to_word=prepare_data(poetrys,64)
	print('***data segmented***')
    print('***Training Begin:')
    train_charrnn=CHARRNN(vocab_size=vocal_size,n_chunk=n_chunk)
    error=train_charrnn.train(x_batches,y_batches,False)
    print('***Training Done***')
    test=CHARRNN(vocab_size=vocal_size,n_chunk=n_chunk,word_to_num=word_to_num,num_to_word=num_to_word)
    poem=test.predict(True)
	print(poem)
    
if __name__='__main__':
    run()

