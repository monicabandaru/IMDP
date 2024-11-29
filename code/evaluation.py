import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import random
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, precision_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinkPrediction():
    def __init__(self, config):
        self.links = [[], [], []]
        sufs = ['_0', '_50', '_100']
        for i, suf in enumerate(sufs):
            with open(config.test_file + suf) as infile:
                for line in infile.readlines():
                    s, t, label = [int(item) for item in line.strip().split()]
                    self.links[i].append([s, t, label])

    def evaluate(self, embedding_matrix):
        test_y = [[], [], []]
        pred_y = [[], [], []]
        pred_label = [[], [], []]
        for i in range(len(self.links)):
            for s, t, label in self.links[i]:
                test_y[i].append(label)
                if(i<21357):
                    pred_y[i].append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
                if pred_y[i][-1] >= 0:
                    pred_label[i].append(1)
                else:
                    pred_label[i].append(0)

        auc = [0, 0, 0]
        target_names = ['class 0', 'class 1']
        nc=[0,0,0]
        for i in range(len(test_y)):
            auc[i] = roc_auc_score(test_y[i], pred_y[i])
            nc[i] = classification_report(test_y[i], pred_label[i], target_names=target_names)
        print(nc[0])
        return auc


    def inference_via_confidence(self,embedding_matrix):
        positive_samples = []


        G = nx.read_edgelist('/home/project/anaconda3/pycharm-community-2023.1.2/differentialprivacy_experimentation/differentialprivacy_experimentation/data/cora/train_0.5')
        # Assuming your graph is stored as an edge list
        edges = [(node1, node2) for node1, node2 in list(G.edges())]
        print("edges generated")
        # Generate positive samples
        for edge in edges:
            positive_samples.append(edge)
        negative_samples = []
        all_nodes = set(node for edge in edges for node in edge)

        # Generate negative samples
        while len(negative_samples) < len(positive_samples):
            node1 = random.choice(list(all_nodes))
            node2 = random.choice(list(all_nodes))

            # Ensure the pair is not an existing edge
            if (node1, node2) not in edges and node1 != node2:
                negative_samples.append((node1, node2))

        # Assuming you have positive_samples and negative_samples lists as mentioned earlier.

        # Combine positive and negative samples into a single dataset
        all_samples = positive_samples + negative_samples
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)

        # Assuming you have embeddings for each node in the graph (stored in a dictionary where node IDs are keys)
        # For example, if your embeddings are stored in an 'embeddings' dictionary:
        # embeddings[node_id] = [embedding_values]

        # Create feature vectors for each sample
        X = []
        for edge in all_samples:
            # model.wv[node_id]
            node1_embedding = embedding_matrix[0][edge[0]]
            node2_embedding = embedding_matrix[1][edge[1]]
            feature_vector = np.concatenate([node1_embedding, node2_embedding])
            X.append(feature_vector)

        X = np.array(X)
        labels = np.array(labels)

        # Assuming you have a binary classifier trained and stored in 'classifier' to predict edge existence.
        # classifier.predict_proba(X) should return the predicted probabilities.
        classifier = LinearRegression().fit(X, labels)
        # Predict probabilities
        predicted_probs = classifier.predict(X)

        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(labels, predicted_probs)

        # Define a threshold for the predicted probabilities to classify edges as positive or negative
        threshold = 0.5  # You can adjust this threshold as needed

        # Convert predicted probabilities to binary labels
        predicted_labels = (predicted_probs > threshold).astype(int)

        # Calculate precision score
        precision = precision_score(labels, predicted_labels)

        print("ROC-AUC Score:", roc_auc)
        print("Precision Score:", precision)

















'''
import numpy as np
import config
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score

class LinkPrediction():
    def __init__(self, config):
        self.links=[]
        with open(config.test_file) as infile:
            for line in infile.readlines():
                s, t, label = [int(item) for item in line.strip().split()]
                self.links.append([s, t, label])
    def evaluate(self,embedding_matrix):
        print(type(self.links))
        print(self.links[0:2])
        test_y = []
        pred_y = []
        pred_label = []
        for i in range(len(self.links)):
                s=self.links[i][0]
                t=self.links[i][1]
                label=self.links[i][2]
                test_y.append(label)
                pred_y.append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
                if pred_y[-1] >= 0:
                    pred_label.append(1)
                else:
                    pred_label.append(0)

        auc = []
        target_names = ['class 0', 'class 1']
        nc = []
        auc = roc_auc_score(test_y, pred_y)
        nc = classification_report(test_y, pred_label, target_names=target_names)
        print(nc)
        macro_f1 = f1_score(test_y,pred_label, average='macro')
        micro_f1 = f1_score(test_y,pred_label, average='micro')
        precision = precision_score(test_y,pred_label, average='macro')
        print("macro-f1 score=",macro_f1)
        print("micro-f1 score=",micro_f1)
        print("precision score=",precision)
        print("auc score=",auc)
        return auc



'''
'''
def node_classification(model, features, labels):
    # Predict the labels using the trained model
    predicted_labels = model.predict(features)

    # Calculate the macro F1 score
    macro_f1 = f1_score(labels, predicted_labels, average='macro')

    # Calculate the micro F1 score
    micro_f1 = f1_score(labels, predicted_labels, average='micro')

    # Calculate the precision score
    precision = precision_score(labels, predicted_labels, average='macro')

    return macro_f1, micro_f1, precision




    # Extract the features from the model
    features = model.get_all_node_features()

    # Create an array to store the predicted probabilities
    predicted_probs = np.zeros(len(test_edges))

    # Iterate over each test edge
    for i, (src, dst) in enumerate(test_edges):
        # Get the features of the source and destination nodes
        src_features = features[src]
        dst_features = features[dst]

        # Combine the features using a desired method (e.g., concatenation)
        combined_features = np.concatenate([src_features, dst_features])

        # Predict the probability of a link using the trained model
        predicted_prob = model.predict_proba(combined_features)
        predicted_probs[i] = predicted_prob

    # Calculate the ROC score
    roc_score = roc_auc_score(test_labels, predicted_probs)

    return roc_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinkPrediction():
    def __init__(self, config):


    def evaluate(self, embedding_matrix):
        test_y = []
        pred_y = []
        pred_label = []
        for i in range(len(self.links)):
            for s, t, label in self.links:
                test_y.append(label)
                pred_y.append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
                #if pred_y[i][-1] >= 0:
                #    pred_label.append(1)
                #else:
                #    pred_label.append(0)

        auc = []
        target_names = ['class 0', 'class 1']
        nc=[]
        for i in range(len(test_y)):
            auc[i] = roc_auc_score(test_y[i], pred_y[i])
            nc[i] = classification_report(test_y[i], pred_label[i], target_names=target_names)
        #print(nc[0])
        return auc



'''