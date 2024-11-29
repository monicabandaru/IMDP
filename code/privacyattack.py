import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

def create_feature_vector(diff_matrix, degree_vector, num_bins=10):
    # Remove diagonal from difference matrix
    diff_values = np.ravel(diff_matrix[np.triu_indices_from(diff_matrix, k=1)])
    # Create bins with equal frequency discretization
    bin_edges = np.histogram_bin_edges(diff_values, num_bins, 'fd')
    # Discretize values into bins
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='onehot-dense', strategy='uniform')
    bin_counts = discretizer.fit_transform(diff_values.reshape(-1, 1))
    # Calculate feature vector
    feature_vector = np.sum(bin_counts, axis=0)
    feature_vector = np.concatenate([feature_vector, degree_vector / np.sum(degree_vector)])
    return feature_vector

def generate_training_data(embeddings, remove_idx, num_bins=10):
    # Remove node from the network and calculate new embedding
    reduced_embeddings = np.delete(embeddings, remove_idx, axis=0)
    reduced_embedding = embeddings[remove_idx]
    # Calculate distance matrices
    diff_matrix = np.abs(embeddings - reduced_embedding) - np.abs(reduced_embeddings - reduced_embedding)
    degree_vector = np.sum(np.abs(embeddings) > 0, axis=1) - 1
    reduced_degree_vector = np.delete(degree_vector, remove_idx)
    # Generate feature vectors and labels
    X = []
    y = []
    for remove_idx_2 in range(embeddings.shape[0]):
        if remove_idx_2 != remove_idx:
            reduced_embeddings_2 = np.delete(reduced_embeddings, remove_idx_2, axis=0)
            reduced_embedding_2 = reduced_embeddings[remove_idx_2]
            diff_matrix_2 = np.abs(reduced_embeddings - reduced_embedding_2) - np.abs(reduced_embeddings_2 - reduced_embedding_2)
            degree_vector_2 = reduced_degree_vector.copy()
            degree_vector_2 = np.delete(degree_vector_2, remove_idx_2)
            feature_vector = create_feature_vector(diff_matrix_2, degree_vector_2, num_bins)
            label = int(remove_idx in np.where(diff_matrix_2 > 0)[0])
            X.append(feature_vector)
            y.append(label)
    return np.array(X), np.array(y)

# Load the social network node embeddings
embeddings = np.load('D:/imp codes/firstpaper/pythonProject-cora - Copy/results/link_prediction/cora/0-gen.npy')

# Select a target node for the membership inference attack
target_idx = 42
target_embedding = embeddings[target_idx]

# Generate training data for the classifier
X, y = generate_training_data(embeddings, target_idx)

# Split the dataset into training and testing subsets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Train a logistic regression classifier to predict neighbor status
clf = LogisticRegression()
clf.fit(train_X, train_y)

# Use the classifier to predict the neighbor status of the target node
target_features = create_feature_vector(np.abs(embeddings - target_embedding) - np.abs(embeddings - embeddings[target_idx]), np.sum(np.abs(embeddings) > 0, axis=1) - 1)
target_prediction = clf.predict([target_features])[0]
print(f'Target node neighbor status: {target_prediction}')
