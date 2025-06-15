from network import Network

NNModel = Network([5, 2, 3, 1])

print("Network's BIASES")
for i, bias in enumerate(NNModel.biases):
    print(f"Bias in Layer {i+1}: (Shape {bias.shape})")
    print(bias)

print("\nNetwork's WEIGHTS")
for i, weights in enumerate(NNModel.weights):
    print(f"Weights in Layer {i+1}: (Shape {weights.shape})")
    print(weights)
