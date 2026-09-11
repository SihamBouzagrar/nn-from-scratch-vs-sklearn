import numpy as np
import matplotlib.pyplot as plt

from src.engine.activations import (
    sigmoid,
    tanh,
    relu,
    identity,
)


x = np.linspace(-5, 5, 500)

plt.figure()
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()


plt.figure()
plt.plot(x, tanh(x))
plt.title("Tanh")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()


plt.figure()
plt.plot(x, relu(x))
plt.title("ReLU")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()


plt.figure()
plt.plot(x, identity(x))
plt.title("Identity")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()