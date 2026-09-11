from src.pipeline.grid_search import GridSearchScratch


grid = GridSearchScratch(
    model_class=None,
    param_grid={
        "learning_rate": [0.01, 0.05],
        "hidden_layer_size": [4, 8],
    },
    cv=None,
)


combinations = (
    grid._generate_param_combinations()
)


print("\n========== GRID ==========\n")

for i, params in enumerate(
    combinations,
    start=1
):
    print(
        f"Configuration {i}: {params}"
    )