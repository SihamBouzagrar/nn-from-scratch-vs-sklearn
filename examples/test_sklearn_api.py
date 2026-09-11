from src.engine.mlp_classifier import (
    MLPClassifierScratch
)


model = MLPClassifierScratch(
    hidden_layer_size=8,
    learning_rate=0.05,
    epochs=3000
)


print("========== PARAMETERS ==========")

print(model.get_params())


print("\n========== CHANGE PARAMETERS ==========")

model.set_params(
    learning_rate=0.01,
    hidden_layer_size=16
)

print(model.get_params())