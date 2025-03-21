import network
import network_matrixbased
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# print("Original sample-by-sample backpropagation")
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 15, 3.0, test_data=test_data)
# # ~ 2.9 seconds per mini-batch
# # ~ Epoch 30: 9489 / 10000

print("Matrix-based backpropagation")
net_matrixbased = network_matrixbased.Network_MatrixBased([784, 30, 10])
net_matrixbased.SGD(training_data, 30, 15, 3.0, test_data=test_data)
# ~ 0.61 seconds per mini-batch
# ~ Epoch 30: 9424 / 10000
