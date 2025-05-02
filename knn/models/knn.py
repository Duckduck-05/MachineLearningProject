import random
import torch
import numpy as np

class KNearestNeighbors():
    def __init__(self, x: np.ndarray, k: int, y: np.ndarray, similarity:str = "euclid", device = "cpu"):
        if device == "cuda":
            self.CUDA = True
        else:
            self.CUDA = False
        self.device = device
        self.x = x # num_samples x num_features
        self.y = y # num_samples x 1
        self.k = k

        if similarity == "cosine":
            self.similarity = self._cosine
        elif similarity == "euclid":
            self.similarity = self._euclid
        else:
            self.similarity = self._manhattan

    def _euclid(self, x1, x2):
        '''Return the Euclidian distance between `x1` and `x2`. `x1.shape` should be `(batch_size, 1, num_features)` while `x2.shape` should be `(num_samples, num_features)`'''
        if self.CUDA: # (x1 - x2).shape: batch_size, num_samples, num_features
            return torch.norm(torch.from_numpy(x1).to(self.device) - torch.from_numpy(x2).to(self.device), dim=-1).cpu().numpy()
        return np.linalg.norm(x1 - x2, axis=-1)
    
    def _manhattan(self, x1, x2):
        if self.CUDA:
            return torch.abs(torch.from_numpy(x1).to(self.device) - torch.from_numpy(x2).to(self.device)).sum(-1).cpu().numpy()
        return np.abs(x1 - x2).sum(-1)

    def _cosine(self, x1, x2):
        '''Return 1 - Cosine similarity between `x1` and `x2`. `x1.shape` should be `(batch_size, 1, num_features)` while `x2.shape` should be `(num_samples, num_features)`'''
        return 1-torch.nn.functional.cosine_similarity(torch.from_numpy(x1).to(self.device), torch.from_numpy(x2).to(self.device), dim=-1).cpu().numpy()
    
    def distance(self, x):
        # Reshape into batch_size x num_features
        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)
        if len(x.shape) == 2: # Batched
            return self.similarity(np.expand_dims(x, 1), self.x)
        else:
            raise ValueError(f"The shape {x.shape} for input is not allowed for training data's shape {self.x.shape}")
    
    def predict(self, sample, reduction="argmax", *, train=False):
        batch_size = 1 if len(sample.shape) == 1 else sample.shape[0]
        distance = self.distance(sample)
        # Get the indices of the top `k` nearest neighbors
        if self.k == 0:
            indices = np.argsort(distance, -1)
        else:
            indices = np.argsort(distance, -1)[:, :self.k]

        if train:
            indices = indices[1:]
        # Get the nearest distance for each sample in the batch
        nearest_distance = np.array([distance[sample_idx][indices[sample_idx]] for sample_idx in range(batch_size)])
        nearest_y = self.y[indices]
        if self.k == 0:
            nearest_distance *= torch.from_numpy(nearest_distance).softmax(-1).numpy()

        # Vote for the class
        vote = np.zeros([batch_size, self.y.max()+1])
        for i in range(self.y.max() + 1):
            vote[:, i] += (1/nearest_distance * (nearest_y == i)).sum(axis=-1)
        if reduction is None:
            return vote
        if reduction == "indices":
            return indices
        if reduction == "score":
            logits = np.argmax(vote, axis=-1)
            score  = np.array([vote[idx][val] for idx, val in enumerate(list(logits))])
            return score / vote.sum(axis=-1), logits
        if reduction == "distance":
            return nearest_distance
        if reduction=="argmax":
            return np.argmax(vote,axis=-1)


class WeightedKNearestNeighbors:
    def __init__(self, x: np.ndarray, k: int, y: np.ndarray, similarity:str = "euclid", weights = 1, learning_rate = 0.01, device = "cpu", train_split_ratio = 0.7):
        self.x = x.detach().to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)
        self.k=k
        self.y=y.detach().to(device) if isinstance(y, torch.Tensor) else torch.tensor(y, device=device)
        similarity = similarity.lower()
        if similarity == "euclid":
            self.similarity = self._euclid
        if similarity == "cosine":
            self.similarity = self._cosine
        if similarity == "manhattan":
            self.similarity = self._manhattan
        self.device=device

        # Init weights
        if isinstance(weights, torch.nn.Module):
            self.weights = weights.to(device)
            self.optimizer = torch.optim.SGD(params=self.weights.parameters(), lr=learning_rate, maximize=True)
        else:
            if isinstance(weights, int): # Init a new weight with shape (in_features, out_features)
                if weights == 0: # Feature weighting mode
                    self.weights = torch.ones([self.x.shape[-1], 1], device=self.device, requires_grad=True)
                else: # Linear transformation
                    self.weights = torch.randn([self.x.shape[-1], weights], device=self.device, requires_grad=True)
            elif isinstance(weights, torch.Tensor):
                if not weights.requires_grad:
                    self.weights = torch.tensor(weights, requires_grad=True).to(device)
                else:
                    self.weights = weights.to(self.device)
            elif isinstance(weights, np.ndarray):
                self.weights = torch.tensor(weights, requires_grad=True, device=self.device)
            self.optimizer = torch.optim.SGD(params=[self.weights], lr=learning_rate, maximize=True)

        self.split_ratio = train_split_ratio
    
    def __random_sequential_split(self, data: torch.Tensor, split_ratio):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        dat_len = len(data)
        start = random.choice(range(int(dat_len * (1-split_ratio))))
        end = start+int(dat_len * split_ratio)
        chosen = data[start:end]
        residual = torch.cat([data[:start], data[end:]], 0)
        return chosen, residual
    
    def __weighting(self, data):
        '''Weights the data. Data is required to be 2D'''
        if isinstance(self.weights, torch.Tensor):
            if len(torch.squeeze(self.weights).shape) == 1: # 1D weight -> element-wise multiplication
                return self.weights.T * data # Weights are inited in shape (180 x 1)
            
            # 2D weight -> Matrix multiplication
            return data @ self.weights
        return self.weights(data)
    
    def _euclid(self, x1, x2):
        '''Return the Euclidian distance between `x1` and `x2`. `x1.shape` should be `(batch_size, 1, num_features)` while `x2.shape` should be `(num_samples, num_features)`'''
        # (x1 - x2).shape: batch_size, num_samples, num_features
        return torch.norm(x1 - x2, dim=-1)
    
    def _manhattan(self, x1, x2):
        return torch.abs(x1 - x2).sum(dim=-1)

    def _cosine(self, x1, x2):
        '''Return 1 - Cosine similarity between `x1` and `x2`. `x1.shape` should be `(batch_size, 1, num_features)` while `x2.shape` should be `(num_samples, num_features)`'''
        return 1-torch.nn.functional.cosine_similarity(x1, x2, dim=-1) + 10**-6
    
    def distance(self, x:torch.Tensor, x2=None, numpy=True) -> np.ndarray|torch.Tensor:
        # Reshape into batch_size x num_features
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x2 is None:
            x2 = self.x
        x2 = x2 if isinstance(x2, torch.Tensor) else torch.tensor(x2)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)
        if len(x.shape) == 2: # Batched
            res = self.similarity(self.__weighting(torch.unsqueeze(x, 1)), self.__weighting(x2))
            return res.detach().cpu().numpy() if numpy else res
        else:
            raise ValueError(f"The shape {x.shape} for input is not allowed for training data's shape {self.x.shape}")
        
    def train(self, epochs=1, batch_size=1, *, verbose=True):
        index = torch.arange(len(self.x))
        for e in range(epochs):
            pred_idx, train_idx = self.__random_sequential_split(index, self.split_ratio)
            pred_perm = torch.randperm(len(pred_idx)) # Shuffle the pred set
            pred_x, pred_y = self.x[pred_idx][pred_perm], self.y[pred_idx][pred_perm]
            train_perm = torch.randperm(len(train_idx)) # Shuffle train set
            train_x, train_y = self.x[train_idx][train_perm], self.y[train_idx][train_perm]
            for idx in range(0, len(train_x), batch_size):
                if len(train_x) - idx < batch_size:
                    x, y = train_x[idx:].to(self.device), train_y[idx:].to(self.device)
                else:
                    x, y = train_x[idx:idx+batch_size].to(self.device), train_y[idx:idx+batch_size].to(self.device)
                distance = self.distance(x, pred_x, numpy=False)

                prob = torch.softmax(1/distance, -1)
                corr_map = pred_y.reshape(1, -1) == y.reshape(-1, 1)
                corr_prob = ((prob * corr_map).sum(axis=-1)).mean()
                # sum(-1): sum for each sample
                # mean(): batch reduction
                if verbose:
                    print(f"Epoch {e + 1} --- Correct probability: {corr_prob.item():.3f}", end='\r')
                corr_prob.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, sample, reduction="argmax"):
        batch_size = 1 if len(sample.shape) == 1 else sample.shape[0]
        distance = self.distance(sample.to(self.device))
        # Get the indices of the top `k` nearest neighbors
        if self.k == 0:
            indices = np.argsort(distance, -1)
        else:
            indices = np.argsort(distance, -1)[:, :self.k]

        # Get the nearest distance for each sample in the batch
        nearest_distance = np.array([distance[sample_idx][indices[sample_idx]] for sample_idx in range(batch_size)])
        nearest_y = self.y[indices].cpu().numpy()
        if self.k == 0:
            nearest_distance *= torch.from_numpy(nearest_distance).softmax(-1).numpy()
        # Vote for the class
        vote = np.zeros([batch_size, self.y.max()+1])
        for i in range(self.y.max() + 1):
            vote[:, i] += (1/nearest_distance * (nearest_y == i)).sum(axis=-1)
        if reduction is None:
            return vote
        if reduction == "indices":
            return indices
        if reduction == "score":
            logits = np.argmax(vote, axis=-1)
            score  = np.array([vote[idx][val] for idx, val in enumerate(list(logits))])
            return score / vote.sum(axis=-1), logits
        if reduction == "distance":
            return nearest_distance