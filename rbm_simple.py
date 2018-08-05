import torch


class RBM_simple():

    def __init__(self, num_visible, num_hidden, learning_rate=1e-3, use_cuda=False):
        """
        :param num_visible: int
            Number of visible nodes
        :param num_hidden: int
            Number of hidden nodes
        :param k:
            step of Gibbs sampling (Contrastive Divergence as optimizer). Usually = 1
        :param learning_rate: float
        :param use_cuda : Boolean
            Using GPU or not
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # self.k = k
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda
        self.train_loss = 0

        self.weights = torch.randn(num_hidden, num_visible)
        self.visible_bias = torch.ones(num_visible)
        self.hidden_bias = torch.ones(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

    def sample_hidden(self, visible_prob):
        """
        :param visible_prob:
            Probability of visible layer
        :return:
            Hidden probability & Sample from hidden prob (Bernoulli distribution)
        """
        hidden_activations = torch.mm(visible_prob, self.weights.t()) + self.hidden_bias
        hidden_prob = torch.sigmoid(hidden_activations)
        return hidden_prob, torch.bernoulli(hidden_prob)

    def sample_visible(self, hidden_prob):
        """
        :param hidden_prob:
            Probability of hidden layer
        :return:
            Visible probability & Samples from visible prob (Bernoulli distribution)
        """
        visible_activations = torch.mm(hidden_prob, self.weights) + self.visible_bias
        visible_prob = torch.sigmoid(visible_activations)
        return visible_prob, torch.bernoulli(visible_prob)

    def train(self, v0, vk, ph0, phk):
        """
        :param v0:
        :param vk:
        :param ph0:
        :param phk:
        :return:
        """
        self.weights += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.visible_bias += torch.sum((v0 - vk), 0)
        self.hidden_bias += torch.sum((ph0 - phk), 0)
        self.train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        return self.train_loss
