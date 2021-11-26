import numpy as np
import pickle, csv
import json, nltk
from random import shuffle
import pickle
import numpy as np
from nltk import *
import sys, math
pklFile = open('w2vModel','rb')
mod=pickle.load(pklFile)
fdata = []
matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


novec=0
tot = 0
with open("train_phrases.csv","r") as f:
  spamreader = csv.reader(f, delimiter=',', quotechar='|')
  for ix,i in enumerate(spamreader):
    twt = i[0].replace(',','')
    lbl = str(i[1])
    
    
    cur_tweet=word_tokenize(twt)[:10]
    w2v_tweet=[]
    for j in cur_tweet:
      tot+=1
      try:
        vec=mod[str(j)]
        w2v_tweet.append(vec)
        
      except Exception:
        novec+=1
        continue
    if len(w2v_tweet)==0:
      continue
    # while len(w2v_tweet)!=10:
      # w2v_tweet+= [np.zeros(32)]
    l=[0,0,0,0,0]
    lbl_list=l[{'very negative':0, 'negative':1, 'neutral':2, 'positive':3, 'very positive':4}[lbl]]=1
    op = [np.array(l)]*len(w2v_tweet) #0-displeasure; 1-compliment; 2-misc
    tup=[w2v_tweet,op]
    # print tup
    # sys.exit(0)
    fdata.append(tup)

full_data=[]
doutput=[]
for i,j in fdata:
  full_data.append(i)
  doutput.append(j)
# print "Length: "+str(len(full_data))
data=full_data[:100000]
output=doutput[:100000]

# print novec
# print tot

with open("test_phrases.csv","r") as f:
  spamreader = csv.reader(f, delimiter=',', quotechar='|')
  for ix,i in enumerate(spamreader):
    twt = i[0]
    lbl = str(i[1])
    
    
    cur_tweet=word_tokenize(twt)[:10]
    w2v_tweet=[]
    for j in cur_tweet:
      try:
        vec=mod[str(j)]
        w2v_tweet.append(vec)
      except Exception:
        continue
    if len(w2v_tweet)==0:
      continue
    # while len(w2v_tweet)!=10:
      # w2v_tweet+= [np.zeros(32)]
    l=[0,0,0,0,0]
    lbl_list=l[{'very negative':0, 'negative':1, 'neutral':2, 'positive':3, 'very positive':4}[lbl]]=1
    op = [np.array(l)]*len(w2v_tweet) #0-displeasure; 1-compliment; 2-misc
    tup=[w2v_tweet,op]
    fdata.append(tup)
    
full_data=[]
doutput=[]
for i,j in fdata:
  full_data.append(i)
  doutput.append(j)

test_data=full_data[:30000]
test_output=doutput[:30000]#doutput



vocab_size = 32 #length of input sequence
# hyperparameters
# print full_data
hidden_size = 64 # size of hidden layer of neurons
learning_rate = 2e-1
output_size=5


# model parameters
Wxhf = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden forward pass
Wxhb = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden backward pass
Whhf = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden forward pass
Whhb = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden backward pass
Whyf = np.random.randn(output_size, hidden_size)*0.01 # hidden to output forward pass
Whyb = np.random.randn(output_size, hidden_size)*0.01 # hidden to output backward pass
bhf = np.zeros((hidden_size, 1)) # hidden bias forward pass
bhb = np.zeros((hidden_size, 1)) # hidden bias backward pass
by = np.zeros((output_size, 1)) # output bias
hprev = np.zeros((hidden_size,1)) # reset RNN memory
hnext = np.zeros((hidden_size,1))
# print Wxhf
# print Whyf
# print "-------------------"
class BiDirectionalRnn:

  def lossFun(self,inputs, targets, hprev, hnext):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hsf, hsb, ys, ps, dy = {}, {}, {}, {}, {}, {}
    hsf[-1] = np.copy(hprev)
    hsb[len(inputs)]=np.copy(hnext)
    loss = 0
    
    # forward pass
    for t in xrange(len(inputs)):
      xs[t]=np.array(inputs[t])
      xs[t]=(np.reshape(xs[t],(32,1))) 
      hsf[t] = np.tanh(np.dot(Wxhf, xs[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state

    for t in reversed(xrange(len(inputs))):
      xs[t]=np.array(inputs[t])# 1 review's each word
      xs[t]=(np.reshape(xs[t],(32,1)))
      hsb[t] = np.tanh(np.dot(Wxhb, xs[t]) + np.dot(Whhb, hsb[t+1]) + bhb) # hidden state
    
    for t in xrange(len(targets)):
      ys[t] = np.dot(Whyf, hsf[t]) + np.dot(Whyb, hsb[t]) #+ by # unnormalized log probabilities for next chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      # loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

    # backward pass: compute gradients going backwards
    
    dWxhf, dWxhb, dWhhf, dWhhb, dWhyf, dWhyb = np.zeros_like(Wxhf), np.zeros_like(Wxhb), np.zeros_like(Whhf), np.zeros_like(Whhb), np.zeros_like(Whyf), np.zeros_like(Whyb)
    dbhf, dbhb, dby = np.zeros_like(bhf), np.zeros_like(bhb), np.zeros_like(by)
    dhfnext = np.zeros_like(hsf[0])
    dhbprev = np.zeros_like(hsb[0])
    
    for t in reversed(xrange(len(inputs))):
      dy = np.copy(ps[t])
      targets[t]=(np.reshape(targets[t],(output_size,1)))
      dy = dy - targets[t] #-------------------------------------
      dWhyf += np.dot(dy, hsf[t].T)
      dWhyb += np.dot(dy, hsb[t].T)
      dby += dy

    for t in reversed(xrange(len(inputs))):
      dh = np.dot(Whyf.T, dy) + dhfnext # backprop into h
      dhraw = (1 - hsf[t] * hsf[t]) * dh # backprop through tanh nonlinearity
      dbhf += dhraw
      dWxhf += np.dot(dhraw, xs[t].T)
      dWhhf += np.dot(dhraw, hsf[t-1].T)
      dhfnext = np.dot(Whhf.T, dhraw)

    for t in xrange(len(inputs)):
      dh = np.dot(Whyb.T, dy) + dhbprev # backprop into h
      dhraw = (1 - hsb[t] * hsb[t]) * dh # backprop through tanh nonlinearity
      dbhb += dhraw
      dWxhb += np.dot(dhraw, xs[t].T)
      dWhhb += np.dot(dhraw, hsb[t+1].T)
      dhbprev = np.dot(Whhf.T, dhraw)

    for dparam in [dWxhf, dWxhb, dWhhf, dWhhb, dWhyf, dWhyb, dbhf, dbhb, dby]:
        np.clip(dparam, -3, 3, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxhf, dWxhb, dWhhf, dWhhb, dWhyf, dWhyb, dbhf, dbhb, dby, hsf[len(inputs)-1], hsb[0]


  def train(self):
    n, p = 0, 0
    mWxhf, mWxhb, mWhhf, mWhhb, mWhyf, mWhyb = np.zeros_like(Wxhf),np.zeros_like(Wxhb), np.zeros_like(Whhf), np.zeros_like(Whhb), np.zeros_like(Whyf), np.zeros_like(Whyb)
    mbhf, mbhb, mby = np.zeros_like(bhf),np.zeros_like(bhb), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocab_size)

    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    hnext = np.zeros((hidden_size,1))
    for epoch in range(3):
      for i in range(len(data)):
        # forward seq_length characters through the net and fetch gradient
        loss, dWxhf, dWxhb, dWhhf, dWhhb, dWhyf, dWhyb, dbhf, dbhb, dby, hprev, hnext = self.lossFun(data[i], output[i], hprev, hnext)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxhf, Wxhb, Whhf, Whhb, Whyf, Whyb, bhf, bhb, by], 
                                      [dWxhf, dWxhb, dWhhf, dWhhb, dWhyf, dWhyb, dbhf, dbhb, dby], 
                                      [mWxhf, mWxhb, mWhhf, mWhhb, mWhyf, mWhyb, mbhf, mbhb, mby]):
          mem += dparam * dparam
          param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
      # print Wxhf
      # print Whyf
  def predict(self):
    cnt=0
    fp=0
    tp=0
    fn=0
    # print(len(test_data))

    for ix,inp in enumerate(test_data):
      op=[]
      xs, hsf, hsb, ys, ps = {} ,{} ,{} ,{} ,{}
      hsf[-1] = np.copy(hprev)
      hsb[len(inp)]=np.copy(hnext)

      #forward
      for t in xrange(len(inp)):
        xs[t] =  np.array(inp[t]) # encode in 1-of-k representation
        xs[t]=(np.reshape(xs[t],(32,1))) 
        hsf[t] = np.tanh(np.dot(Wxhf, xs[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state

      for t in reversed(xrange(len(inp))):
        xs[t] =  np.array(inp[t]) # encode in 1-of-k representation
        xs[t]=(np.reshape(xs[t],(32,1)))
        hsb[t] = np.tanh(np.dot(Wxhb, xs[t]) + np.dot(Whhb, hsb[t+1]) + bhb) # hidden state
          
      for t in xrange(len(inp)):
        ys[t] = np.dot(Whyf, hsf[t]) + np.dot(Whyb, hsb[t]) # + by  unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      # print ps

      pos=0.0
      verypos=0.0
      neg=0.0
      veryneg=0.0
      neutral=0.0

      dicCount = {0:0, 1:0, 2:0, 3:0, 4:0}
      for val in ps.values():
        # print val
        index = np.argmax(val)
        dicCount[index]+=1

      # print dicCount
      
      pred = np.argmax(dicCount.values())



      # for t in range(len(ps)):
      #   veryneg+=ps[t][0]
      #   neg+=ps[t][1]
      #   neutral+=ps[t][2]
      #   pos+=ps[t][3]
      #   verypos+=ps[t][4]

      # veryneg +=ps[0][0]
      # neg +=ps[0][1]
      # neutral+=ps[0][2]
      # pos +=ps[0][3]
      # verypos +=ps[0][4]



      # neg=neg/len(ps)
      # veryneg=veryneg/len(ps)
      # pos/=len(ps)
      # verypos/=len(ps)
      # neutral/=len(ps)
      # print(pos,neg_data)
      # maxval=max(pos,neg,neutral,verypos,veryneg)
      # if maxval==veryneg:
      #   pred=0
      # elif maxval==neg:
      #   pred=1
      # elif maxval==neutral:
      #   pred=2
      # elif maxval==pos:
      #   pred=3
      # else:
      #   pred=4

      if test_output[ix][0][0] == 1:
        actual=0
      elif test_output[ix][0][1] == 1:
        actual=1
      elif test_output[ix][0][2] == 1:
        actual=2
      elif test_output[ix][0][3] == 1:
        actual=3
      else:
        actual=4
      
      matrix[actual][pred]+=1
      # print matrix
      # print "Actual: "+str(actual)
      # print "Predicted: "+str(pred)
      if pred==actual:
        cnt+=1
    # print dicCount  
    print matrix
    # print "Class: Very negative:-"
    # print "Recall: "+str(100.0*float(matrix[0][0])/sum(matrix[0]))
    # print "Precision: "+str(100.0*float(matrix[0][0])/sum([val[0] for val in matrix]))

    # print "Class: Negative:-"
    # print "Recall: "+str(100.0*float(matrix[1][1])/sum(matrix[1]))
    # print "Precision: "+str(100.0*float(matrix[1][1])/sum([val[1] for val in matrix]))

    # print "Class: Neutral:-"
    # print "Recall: "+str(100.0*float(matrix[2][2])/sum(matrix[2]))
    # print "Precision: "+str(100.0*float(matrix[2][2])/sum([val[2] for val in matrix]))

    # print "Class: Positive:-"
    # print "Recall: "+str(100.0*float(matrix[3][3])/sum(matrix[3]))
    # print "Precision: "+str(100.0*float(matrix[3][3])/sum([val[3] for val in matrix]))

    # print "Class: Very positive:-"
    # print "Recall: "+str(100.0*float(matrix[4][4])/sum(matrix[4]))
    # print "Precision: "+str(100.0*float(matrix[4][4])/sum([val[4] for val in matrix]))
    print('Accuracy: '+str(cnt*100.0/len(test_data)))   


  def predictUser(self):
    while True:
      inp1 = raw_input("Enter text: ")
      
      inp=[]
      for i in word_tokenize(inp1):
        try:
          inp.append(mod[i])
        except Exception:
          pass



      if len(inp)==0:
        return 0
      xs, hsf, hsb, ys, ps = {} ,{} ,{} ,{} ,{}
      hsf[-1] = np.copy(hprev)
      hsb[len(inp)]=np.copy(hnext)
      for t in xrange(len(inp)):
          xs[t] =  np.array(inp[t]) # encode in 1-of-k representation
          xs[t]=(np.reshape(xs[t],(32,1))) 
          hsf[t] = np.tanh(np.dot(Wxhf, xs[t]) + np.dot(Whhf, hsf[t-1]) + bhf) # hidden state

      for t in reversed(xrange(len(inp))):
        xs[t] =  np.array(inp[t]) # encode in 1-of-k representation
        xs[t]=(np.reshape(xs[t],(32,1)))
        hsb[t] = np.tanh(np.dot(Wxhb, xs[t]) + np.dot(Whhb, hsb[t+1]) + bhb) # hidden state
          
      for t in xrange(len(inp)):
        ys[t] = np.dot(Whyf, hsf[t]) + np.dot(Whyb, hsb[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars # probabilities for next chars
        # print np.argmax(ps[t]) 
      
      pos=0.0
      verypos=0.0
      neg=0.0
      veryneg=0.0
      neutral=0.0

      dicCount = {0:0, 1:0, 2:0, 3:0, 4:0}
      for val in ps.values():
        # print val
        index = np.argmax(val)
        dicCount[index]+=1

      # print dicCount
      
      pred = np.argmax(dicCount.values())
      print pred