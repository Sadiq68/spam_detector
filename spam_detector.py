import numpy
import re
import math
#import glob
import os
#from nb_model import NB_model 

class NB_model:

    def __init__(self,msg):
        self.spam_prob_dict ={}
        self.ham_prob_dict = {}
        self.prob_spam = 0
        self.prob_ham = 0
        self.tot_ham_words = 0
        self.tot_spam_words = 0
        #print(msg)

    def getData(self):
        vocab = []
        hamwords = {}
        hamdoc_count = 0

        for file in os.listdir('./train/'):
            if (file.startswith('train-ham')):
                hamdoc_count  = hamdoc_count+1
                file = "./train/"+file
                doc = open(file, 'r').read().lower()
                doc = re.sub('[a-z]*@*.com', '', doc)
                doc = re.sub('<*[a-z]*>', '', doc)
                words = [word for word in re.split('[^a-zA-Z]',doc)]
                for word in words:
                    if word != '':
                        try:
                            hamwords[word] += 1
                        except KeyError:
                            hamwords[word] = 1

        spamdoc_count = 0
        spamwords = {}

        for file in os.listdir('./train/'):
            if (file.startswith('train-spam')):
                file = "./train/"+file
                spamdoc_count = spamdoc_count+1
                doc = open(file, 'r').read().lower()
                words = [word for word in re.split('[^a-zA-Z]',doc)]
                for word in words:
                    if word != '':
                        try:
                            spamwords[word] += 1
                        except KeyError:
                            spamwords[word] = 1
        return hamwords, spamwords, hamdoc_count, spamdoc_count

    def train(self, hm_dict, sp_dict, hm_prob, sp_prob):
        vocab_set = []
        self.prob_spam = sp_prob
        self.prob_ham = hm_prob
        for key,value in hm_dict.items():
            self.tot_ham_words += value
            vocab_set.append(key)
        vocab_set = set(vocab_set)

        print(len(vocab_set))
        for key,value in spamwords.items():
            self.tot_spam_words += value
            vocab_set.add(key)
        print(len(vocab_set))
        self.vocab = list(vocab_set)
        print(len(self.vocab))
        file = open("./model.txt", 'w+')

        index = 1
        for word in sorted(self.vocab):
            try:
                hm_dict[word] += 0
            except KeyError:
                hm_dict[word] = 0

            try:
                sp_dict[word] += 0
            except KeyError:
                sp_dict[word] = 0

            prob_word_ham = (hm_dict[word]+0.5)/(self.tot_ham_words+len(self.vocab))
            self.ham_prob_dict[word] = prob_word_ham
            prob_word_spam = (sp_dict[word]+0.5)/(self.tot_spam_words+len(self.vocab))
            self.spam_prob_dict[word] = prob_word_spam
           # print("%d  %s  %d  %f  %d  %f"%(index, word, hm_dict[word], prob_word_ham, sp_dict[word], prob_word_spam))
            file.write("%d  %s  %d  %f  %d  %f\n"%(index, word, hm_dict[word], prob_word_ham, sp_dict[word], prob_word_spam))
            index +=1

        file.close()

    def test(self, words):
        score_ham = math.log10(self.prob_ham)
        score_spam = math.log10(self.prob_spam)
        for word in words:
            if word != '':
                try:
                    self.ham_prob_dict[word] += 0
                except KeyError:
                    self.ham_prob_dict[word] = 0.5/(self.tot_ham_words+len(self.vocab))
                    self.spam_prob_dict[word] = 0.5/(self.tot_spam_words+len(self.vocab))

                score_ham = score_ham + math.log10(self.ham_prob_dict[word])
                score_spam = score_spam + math.log10(self.spam_prob_dict[word])

        if score_ham > score_spam:
            return("ham",score_ham,score_spam)
        else:
            return("spam",score_ham,score_spam)

model = NB_model("creating object")
hamwords, spamwords, hamdoc_count, spamdoc_count = model.getData()
prob_ham = (hamdoc_count/(hamdoc_count+spamdoc_count))
prob_spam = (spamdoc_count/(hamdoc_count+spamdoc_count))
model.train(hamwords,spamwords,prob_ham,prob_spam)

index = 1
y_act = []
y_pred = []
res_file = open("./result.txt", 'w+')

for file in os.listdir('./test/'):
    if (file.startswith('test-ham')):
        file = "./test/"+file
        doc = open(file, 'r').read().lower()
        words = [word for word in re.split('[^a-zA-Z]',doc)]
        (result,h_score,s_score) = model.test(words)
        y_act.append(0)
        if result == "ham":
            output = 'right'
            y_pred.append(0)
        else:
            output = 'wrong'
            y_pred.append(1)
    #print("%d  %s  %s  %f  %f  %s  %s"%(index, file, result, h_score, s_score, "ham", output))
        res_file.write("%d  %s  %s  %f  %f  %s  %s \n"%(index, file, result, h_score, s_score, "ham", output))
        index += 1

for file in os.listdir('./test/'):
    if (file.startswith('test-spam')):
        file = "./test/"+file
        doc = open(file, 'r', encoding='cp437').read().lower()
        words = [word for word in re.split('[^a-zA-Z]',doc)]
        (result,h_score,s_score) = model.test(words)
        y_act.append(1)
        if result == "ham":
            output = 'wrong'
            y_pred.append(0)
        else:
            output = 'right'
            y_pred.append(1)
    #print("%d  %s  %s  %f  %f  %s  %s"%(index, file, result, h_score, s_score, "spam", output))
        res_file.write("%d  %s  %s  %f  %f  %s  %s \n"%(index, file, result, h_score, s_score, "spam", output))
        index += 1

res_file.close()


def resultFn(true_pos, true_neg, false_pos, false_neg):
    print("-------------------------------------------Evaluation--------------------------------------------")
    print("------------------------------------------")
    print(" | "+"true_pos : "+ str(true_pos)+" | "+"false_pos : "+ str(false_pos)+" | " +"\n | false_neg : "+ str(false_neg) +" | "+"true_neg : "+str(true_neg)+" | ")
    print("------------------------------------------")
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2 * ((precision * recall)/(precision + recall))
    accuracy = (true_pos)/(true_pos + false_neg)
    print("Precision : "+str(precision*100) + "\nRecall : "+str(recall*100)+"\nF1-score : "+str(f1*100)+"\nAccuracy : "+str(accuracy*100))

# positive = ham and negative = spam
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(y_act)):
    if y_act[i] == 1 and y_pred[i] == 1:
        tp += 1
    elif y_act[i] == 1 and y_pred[i] == 0:
        fn += 1
    elif y_act[i] == 0 and y_pred[i] == 1:
        fp += 1
    else:
        tn += 1
print("HAM Results")
resultFn(tp, tn, fp, fn)

# positive = spam and negative = ham
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(y_act)):
    if y_act[i] == 0 and y_pred[i] == 0:
        tp += 1
    elif y_act[i] == 0 and y_pred[i] == 1:
        fn += 1
    elif y_act[i] == 1 and y_pred[i] == 0:
        fp += 1
    else:
        tn += 1
print("SPAM Results")
resultFn(tp, tn, fp, fn)

print("Total Accuracy : %f"%(100*(tp+tn)/(tp+tn+fp+fn)))
