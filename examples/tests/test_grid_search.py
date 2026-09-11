from src.pipeline.grid_search import GridSearchScratch


def test_grid_generation():

    grid = GridSearchScratch(
        model_class=None,
        param_grid={
            "learning_rate": [0.01, 0.1],
            "hidden_layer_size": [4, 8],
        },
        cv=None,
    )

    combinations = (
        grid._generate_param_combinations()
    )

    assert len(combinations) == 4