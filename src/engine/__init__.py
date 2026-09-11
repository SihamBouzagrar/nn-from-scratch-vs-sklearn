def __init__(
    self,
    hidden_layer_size=2,
    learning_rate=0.1,
    epochs=1000,
    random_state=None,
):
    self.hidden_layer_size = hidden_layer_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.random_state = random_state

    self.weights_input_hidden = None
    self.bias_hidden = None

    self.weights_hidden_output = None
    self.bias_output = None