import matplotlib.pyplot as plt
import numpy as np


class SLP(object):
    
    def __init__(self, iris, epoch, kfold, learn_rate):
        self.iris = iris
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.kfold = kfold
        self.weights = [0.25,0.5,0.75,1] #np.random.uniform(low=0, high=1, size=(4))
        self.bias = 0.5 #np.random.uniform(low=0, high=1, size=(1))
        self.error_training = 0
        self.error_test = 0
        self.hit_training = 0
        self.hit_test = 0
        self.accuracy_training = 0
        self.accuracy_test = 0
        self.all_error_training = []
        self.all_accuracy_training = []
        self.all_error_test = []
        self.all_accuracy_test = []
        self.kfold_split()

    def sigmoid(self, result):
        return 1 / (1 + np.exp(-result))
    
    def prediction(self, sigmoid):
        if sigmoid <= 0.5:
            return 0
        else:
            return 1
    
    def update_bias(self, data, sigmoid):
        target = data[-1]
        self.bias = 2*(sigmoid-target)*(1-sigmoid)*sigmoid
        #print(self.bias)
    
    def update_weights(self, data, bias):
        old_weights = self.weights.copy()
        for index, _ in enumerate(self.weights):
            self.weights[index] = old_weights[index] - (self.learn_rate*bias*data[index])
        print(self.weights)

    def train_data(self, iris):
        for data in iris:
            result = np.dot(data[:4], self.weights)
            sigmoid = self.sigmoid(result)
            error = pow(data[-1]-sigmoid,2)
            self.error_training += error
            predict = int(self.prediction(sigmoid))
            if predict == data[-1]:
                self.hit_training += 1
            self.update_bias(data, sigmoid)
            self.update_weights(data, self.bias)

    def test_data(self, iris):
        for data in iris:
            result = np.dot(data[:4], self.weights)
            sigmoid = self.sigmoid(result)
            error = pow(data[-1]-sigmoid,2)
            self.error_test += error
            predict = int(self.prediction(sigmoid))
            if predict == data[-1]:
                self.hit_test += 1
            
    def single_run(self):
        self.error_training = 0
        self.error_test = 0
        self.accuracy_training = 0
        self.accuracy_test = 0
        for index in range(self.kfold):
            self.hit_training = 0
            self.hit_test = 0
            self.train_data(self.kfold_train_data[index])
            self.test_data(self.kfold_test_data[index])
            self.accuracy_training += self.hit_training/80
            self.accuracy_test += self.hit_test/20
        self.all_error_training.append(self.error_training/self.kfold)
        self.all_error_test.append(self.error_test/self.kfold)
        self.all_accuracy_training.append(self.accuracy_training/self.kfold*100)
        self.all_accuracy_test.append(self.accuracy_test/self.kfold*100)
        
    def kfold_split(self):
        fold = []
        fold_n = int(len(self.iris)/self.kfold)
        for index in range(self.kfold):
            fold.append(self.iris[index*fold_n:(index+1)*fold_n])
        self.kfold_train_data = []
        self.kfold_test_data = []
        for index in range(self.kfold):
            self.kfold_test_data.append(fold[index])
            train_tmp = []
            for not_index in [x for x in range(self.kfold) if x != index]:
                train_tmp.extend(fold[not_index])
            self.kfold_train_data.append(train_tmp)

    def plot(self, error_training, error_test, accuracy_training, accuracy_test):  
        plt.figure(1)
        plt.plot(error_training, color='green', linewidth=2, label = 'training')
        plt.plot(error_test, color='yellow', linewidth=2, label = 'test')
        plt.title('Sum Error: Learning Rate ' + str(self.learn_rate))
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        
        plt.figure(2)
        plt.plot(accuracy_training, color='green', linewidth=2, label = 'training')
        plt.plot(accuracy_test, color='yellow', linewidth=2, label = 'test')
        plt.title('Accuracy: Learning Rate ' + str(self.learn_rate))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()

    def run(self):
       # np.random.shuffle(self.iris) uncomment untuk shuffle csv row
        for _ in range(0, self.epoch+1):
            self.cur_epoch = _
            self.single_run()
        self.plot(self.all_error_training, self.all_error_test, self.all_accuracy_training, self.all_accuracy_test)
