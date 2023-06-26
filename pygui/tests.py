import unittest
import torch
from model import Net, ConvNet
from trainer import load_data, train, test

class TestModel(unittest.TestCase):
    def setUp(self):
        self.net = Net()
        self.convNet = ConvNet()
        self.random_input_dense = torch.randn(1, 28*28)
        self.random_input_conv = torch.randn(1, 1, 28, 28)

    def test_Net(self):
        # Test the output size of the forward pass
        output = self.net(self.random_input_dense)
        self.assertEqual(output.size(), (1, 10))

        # Test whether model parameters are updated after one step of training
        parameters_before = [param.clone() for param in self.net.parameters()]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        output = self.net(self.random_input_dense)
        loss = criterion(output, torch.tensor([7]))
        loss.backward()
        optimizer.step()
        parameters_after = list(self.net.parameters())
        for p_before, p_after in zip(parameters_before, parameters_after):
            self.assertFalse(torch.equal(p_before, p_after))

    def test_ConvNet(self):
        # Test the output size of the forward pass
        output = self.convNet(self.random_input_conv)
        self.assertEqual(output.size(), (1, 10))

        # Test whether model parameters are updated after one step of training
        parameters_before = [param.clone() for param in self.convNet.parameters()]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.convNet.parameters(), lr=0.001, momentum=0.9)
        output = self.convNet(self.random_input_conv)
        loss = criterion(output, torch.tensor([7]))
        loss.backward()
        optimizer.step()
        parameters_after = list(self.convNet.parameters())
        for p_before, p_after in zip(parameters_before, parameters_after):
            self.assertFalse(torch.equal(p_before, p_after))

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.train_loader, self.test_loader = load_data()

    def test_load_data(self):
        # Check that data was loaded correctly
        self.assertGreater(len(self.train_loader.dataset), 0)
        self.assertGreater(len(self.test_loader.dataset), 0)

    def test_train(self):
        # Test the training function on one epoch
        train_losses = train(self.train_loader, self.test_loader, epochs=1, model_type='simple')
        self.assertTrue(isinstance(train_losses, list))
        self.assertGreater(len(train_losses), 0)

    def test_test(self):
        # Test the test function
        test_loss, test_accuracy = test(self.test_loader, self.net)
        self.assertTrue(isinstance(test_loss, float))
        self.assertTrue(isinstance(test_accuracy, float))

if __name__ == "__main__":
    unittest.main()

