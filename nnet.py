import sys
import re as pattern
import random
import math

# learning rate
learning_rate = float(sys.argv[1])
# the number of hidden units
hidden_units = int(sys.argv[2])
# the number of training epochs
epoch = int(sys.argv[3])
# training data file
train = open(sys.argv[4])
# testing data file
test = open(sys.argv[5])

# list of dictionary:  different features : len(value)=len(type), each inner list includes all value for one feature
# [ {..}, {..}, {..}, {..}, {..}, {..}, {..}, {..} ]
list_of_feature_dict = []
# value dictionary for different labels : ['+', '-']
class_dict = {}
# list of training data (features): for discrete data, each value has one dimension(1 or 0) to achieve one-of-K encoding
train_data = []
# list of training data (labels)
train_data_label = []
# list of testing data (features): for discrete data, each value has one dimension(1 or 0) to achieve one-of-K encoding
test_data = []
# list of testing data (labels)
test_data_label = []
# first layer weight without hidden layer
V0 = []
# first layer weight
V1 = []
# second layer weight
W = []
# evaluation
TP = 0
FP = 0
FN = 0
precision = 0
recall = 0
F1_score = 0
count = 0


def calculate_dot_product(a, b):
    total = 0.0
    for i in range(0, len(a)):
        total += a[i] * b[i]
    return total


def calculate_cross_entropy(output, true_label):
    return -true_label * math.log(output) - (1 - true_label) * math.log(1 - output)


# parse train data
data_line = False
for line in train:
    if pattern.findall('@data', line, pattern.I) != []:
        data_line = True
    elif pattern.findall('@attribute', line, pattern.I) != []:
        line = line.lstrip(' ')
        line = line.rstrip('\n')
        line = line.rstrip('\r')
        line = line.rstrip(' ')
        line = line.split(None, 2)
        line[1] = line[1].strip('\'')
        line[1] = line[1].replace(' ', '')
        line[2] = line[2].strip('{')
        line[2] = line[2].strip('}')
        line[2] = line[2].replace(' ', '')
        line[2] = line[2].split(',')
        if line[1] != 'class':
            # make a dictionary for each feature
            tempp = {}
            # discrete feature
            if len(line[2]) > 1:
                # give each discrete value a number
                for i in range(0, len(line[2])):
                    tempp[line[2][i]] = i
            # numerical feature
            else:
                tempp[line[2][0]] = 'numeric value'
            list_of_feature_dict.append(tempp)
        else:
            class_dict[line[2][0]] = 0
            class_dict[line[2][1]] = 1
    elif data_line is True:
        line = line.strip('\n')
        line = line.strip('\r')
        line = line.replace(' ', '')
        line = line.split(',')
        temp = []
        for i in range(0, len(line) - 1):
            # discrete feature
            if len(list_of_feature_dict[i]) > 1:
                # one-of-K encoding for discrete feature
                for j in range(0, len(list_of_feature_dict[i])):
                    if j == list_of_feature_dict[i][line[i]]:
                        temp.append(1)
                    else:
                        temp.append(0)
            # numerical feature
            else:
                temp.append(float(line[i]))
        train_data.append(temp)
        train_data_label.append(class_dict[line[len(line) - 1]])
    else:
        pass
# add bias neuron for input layer
train_data_with_bias = []
for i in range(0, len(train_data)):
    train_data_with_bias.append([1] + train_data[i])

# parse test data
data_line = False
for line in test:
    if (pattern.findall('@data', line, pattern.I) != []):
        data_line = True
    elif data_line == True:
        line = line.strip('\n')
        line = line.strip('\r')
        line = line.strip('\n')
        line = line.replace(' ', '')
        line = line.split(',')
        temp = []
        for i in range(0, len(line) - 1):
            # discrete feature
            if len(list_of_feature_dict[i]) > 1:
                for j in range(0, len(list_of_feature_dict[i])):
                    if j == list_of_feature_dict[i][line[i]]:
                        temp.append(1)
                    else:
                        temp.append(0)
            # numerical feature
            else:
                temp.append(float(line[i]))
        test_data.append(temp)
        test_data_label.append(class_dict[line[len(line) - 1]])
    else:
        pass
# add bias neuron for input layer
test_data_with_bias = []
for i in range(0, len(test_data)):
    test_data_with_bias.append([1] + test_data[i])


# after one-of-K encode, each instance has more feature dimensions because discrete feature will have more than one slot
# determine the exact index/location for numeric feature
index_of_numeric_feature = []
for i in range(0, len(list_of_feature_dict)):
    # discrete feature
    if len(list_of_feature_dict[i]) > 1:
        count += len(list_of_feature_dict[i])
    # numerical feature
    else:
        # index 0 is prepared for bias neuron
        count += 1
        index_of_numeric_feature.append(count)

# Standardizing all numeric features
mean = [0.0] * len(train_data_with_bias[0])    # determining by training data
sigma = [0.0] * len(train_data_with_bias[0])    # determining by training data
# traverse each numeric feature
for i in range(0, len(index_of_numeric_feature)):
    # traverse each instance
    for j in range(0, len(train_data)):
        mean[index_of_numeric_feature[i]] += train_data_with_bias[j][index_of_numeric_feature[i]]
    mean[index_of_numeric_feature[i]] = mean[index_of_numeric_feature[i]] / len(train_data)
    # traverse each instance
    for j in range(0, len(train_data)):
        difference = train_data_with_bias[j][index_of_numeric_feature[i]] - mean[index_of_numeric_feature[i]]
        sigma[index_of_numeric_feature[i]] += difference * difference
    sigma[index_of_numeric_feature[i]] = math.sqrt(sigma[index_of_numeric_feature[i]] / len(train_data))
    # traverse each training instance
    for j in range(0, len(train_data_with_bias)):
        train_data_with_bias[j][index_of_numeric_feature[i]] = \
            (train_data_with_bias[j][index_of_numeric_feature[i]] - mean[index_of_numeric_feature[i]]) / sigma[index_of_numeric_feature[i]]
    # traverse each testing instance
    for j in range(0, len(test_data_with_bias)):
        test_data_with_bias[j][index_of_numeric_feature[i]] = \
            (test_data_with_bias[j][index_of_numeric_feature[i]] - mean[index_of_numeric_feature[i]]) / sigma[index_of_numeric_feature[i]]

# All weights and bias parameters are initialized to random values in [-0.01, 0.01]
for i in range(0, len(train_data_with_bias[0])):
    V0.append(random.uniform(-0.01, 0.01))

for i in range(0, hidden_units):
    V1.append([])
    for j in range(0, len(train_data_with_bias[0])):
        V1[i].append(random.uniform(-0.01, 0.01))

for i in range(0, hidden_units + 1):
    W.append(random.uniform(-0.01, 0.01))

# Stochastic gradient descent
instance_index = list(range(0, len(train_data_with_bias)))
random.shuffle(instance_index)

# train neural network
input_of_final_layer = 0
output_of_final_layer = 0
if hidden_units == 0:  # no hidden layer
    for i in range(0, epoch):
        # training by SGD
        for j in range(0, len(instance_index)):
            # forward
            current_instance = train_data_with_bias[instance_index[j]]
            input_of_final_layer = calculate_dot_product(current_instance, V0)
            output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))
            # back-propagation update
            y = train_data_label[instance_index[j]]
            for m in range(0, len(train_data_with_bias[0])):
                V0[m] = V0[m] - learning_rate * (output_of_final_layer - y) * current_instance[m]

        # After each training epoch, test on training data
        cross_entropy_error = 0
        correct_classified = 0
        misclassified = 0
        for j in range(0, len(instance_index)):
            current_instance = train_data_with_bias[j]
            input_of_final_layer = calculate_dot_product(current_instance, V0)
            output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))
            cross_entropy_error += calculate_cross_entropy(output_of_final_layer, train_data_label[j])
            if output_of_final_layer <= 0.5:
                prediction = 0
                if prediction == train_data_label[j]:
                    correct_classified = correct_classified + 1
                else:
                    misclassified = misclassified + 1
            else:
                prediction = 1
                if prediction == train_data_label[j]:
                    correct_classified = correct_classified + 1
                else:
                    misclassified = misclassified + 1

        # print each epoch result
        print("%d\t%s\t%d\t%d" % (i + 1, str(cross_entropy_error), correct_classified, misclassified))

elif hidden_units > 0:  # with hidden layer
    for i in range(0, epoch):
        # training by SGD
        for j in range(0, len(instance_index)):
            # forward
            current_instance = train_data_with_bias[instance_index[j]]
            input_of_hidden_layer = []
            output_of_hidden_layer = []
            for k in range(0, hidden_units):
                input_of_hidden_layer.append(calculate_dot_product(current_instance, V1[k]))
                output_of_hidden_layer.append(1 / (1 + math.exp(-input_of_hidden_layer[k])))
            output_of_hidden_layer_with_bias = [1] + output_of_hidden_layer
            input_of_final_layer = calculate_dot_product(output_of_hidden_layer_with_bias, W)
            output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))

            # back-propagation update
            # useoriginal W toupdateV1: loss->final output->final input->hidden output->hidden input->first layer weight
            # loss->final output->final input: (output_of_final_layer - y)
            # final input->hidden output: W[k + 1] except bias in hidden layer
            # hidden output->hidden input: output_of_hidden_layer[k] * (1 - output_of_hidden_layer[k])
            # hidden input->first layer weight: current_instance[l]
            y = train_data_label[instance_index[j]]
            old_W = W
            # update hidden layer weight
            for k in range(0, hidden_units + 1):
                W[k] = W[k] - learning_rate * (output_of_final_layer - y) * output_of_hidden_layer_with_bias[k]
            # update input layer weight
            for k in range(0, hidden_units):
                for l in range(0, len(train_data_with_bias[0])):
                    V1[k][l] = V1[k][l] - learning_rate * (output_of_final_layer - y) * old_W[k + 1] * output_of_hidden_layer[k] * (1 - output_of_hidden_layer[k]) * current_instance[l]

        # After each training epoch, test on training data
        cross_entropy_error = 0
        correct_classified = 0
        misclassified = 0
        for j in range(0, len(train_data_with_bias)):
            current_instance = train_data_with_bias[j]
            input_of_hidden_layer = []
            output_of_hidden_layer = []
            for k in range(0, hidden_units):
                input_of_hidden_layer.append(calculate_dot_product(current_instance, V1[k]))
                output_of_hidden_layer.append(1 / (1 + math.exp(-input_of_hidden_layer[k])))
            output_of_hidden_layer_with_bias = [1] + output_of_hidden_layer
            input_of_final_layer = calculate_dot_product(output_of_hidden_layer_with_bias, W)
            output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))
            cross_entropy_error = cross_entropy_error + calculate_cross_entropy(output_of_final_layer, train_data_label[j])
            if output_of_final_layer > 0.5:
                prediction = 1
            else:
                prediction = 0
            if prediction == train_data_label[j]:
                correct_classified = correct_classified + 1
            else:
                misclassified = misclassified + 1
        print("%d\t%s\t%d\t%d" % (i + 1, str(cross_entropy_error), correct_classified, misclassified))
else:
    print("Value of hidden_units should not be negative.")

# test neural network
input_of_hidden_layer = [0] * hidden_units
output_of_hidden_layer = [0] * hidden_units
if hidden_units == 0:  # no hidden layer
    cross_entropy_error = 0
    correct_classified = 0
    misclassified = 0
    for j in range(0, len(test_data_with_bias)):
        current_instance = test_data_with_bias[j]
        input_of_final_layer = calculate_dot_product(current_instance, V0)
        output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))
        cross_entropy_error += calculate_cross_entropy(output_of_final_layer, test_data_label[j])
        if output_of_final_layer <= 0.5:
            prediction = 0
            if prediction == test_data_label[j]:
                correct_classified = correct_classified + 1
            else:
                misclassified = misclassified + 1
                FN += 1
        else:
            prediction = 1
            if prediction == test_data_label[j]:
                correct_classified = correct_classified + 1
                TP += 1
            else:
                misclassified = misclassified + 1
                FP += 1
        print("%s\t%d\t%d" % (str(output_of_final_layer), prediction, test_data_label[j]))

    # calculate F1-score: (2*(precision*recall))/(precision + recall)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1_score = (2*(precision*recall))/(precision + recall)
    print("%d\t%d" % (correct_classified, misclassified))
    print(str(F1_score))

elif hidden_units > 0:  # with hidden layer
    cross_entropy_error = 0
    correct_classified = 0
    misclassified = 0
    for j in range(0, len(test_data_with_bias)):
        for k in range(0, hidden_units):
            input_of_hidden_layer[k] = calculate_dot_product(test_data_with_bias[j], V1[k])
            output_of_hidden_layer[k] = 1 / (1 + math.exp(-input_of_hidden_layer[k]))
        output_of_hidden_layer_with_bias = [1] + output_of_hidden_layer
        input_of_final_layer = calculate_dot_product(output_of_hidden_layer_with_bias, W)
        output_of_final_layer = 1 / (1 + math.exp(-input_of_final_layer))
        if output_of_final_layer <= 0.5:
            prediction = 0
            if prediction == test_data_label[j]:
                correct_classified = correct_classified + 1
            else:
                misclassified = misclassified + 1
                FN += 1
        else:
            prediction = 1
            if prediction == test_data_label[j]:
                correct_classified = correct_classified + 1
                TP += 1
            else:
                misclassified = misclassified + 1
                FP += 1
        print("%s\t%d\t%d" % (str(output_of_final_layer), prediction, test_data_label[j]))

    # calculate F1-score: (2*(precision*recall))/(precision + recall)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2 * (precision * recall)) / (precision + recall)
    print("%d\t%d" % (correct_classified, misclassified))
    print(str(F1_score))

else:
    print("Value of hidden_units should not be negative.")

