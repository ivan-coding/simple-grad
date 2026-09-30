import random
from pathlib import Path

import config
from simplegrad import MLP, Value, sum_squared_error
from simplegrad.visualize import render_graph


def render_loss_graph(loss: Value, step: int) -> None:
    output_path = Path(config.GRAPH_OUTPUT_DIR) / f"loss_graph_step_{step}"
    render_graph(loss, output_path)
    print(f"Saved loss graph to {output_path}.png")


def train() -> MLP:
    random.seed(config.RANDOM_SEED)
    input_size = len(config.TRAINING_INPUTS[0])
    model = MLP(
        input_size,
        [*config.HIDDEN_LAYER_SIZES, config.OUTPUT_SIZE],
        config.WEIGHT_INIT_RANGE,
    )
    print(f"Model has {len(model.parameters())} parameters")

    first_step, last_step = 1, config.TRAINING_STEPS
    for step in range(first_step, last_step + 1):
        predictions = [model(inputs) for inputs in config.TRAINING_INPUTS]
        loss = sum_squared_error(predictions, config.TRAINING_TARGETS)

        model.zero_grad()
        loss.backward()

        if config.RENDER_LOSS_GRAPHS and step in (first_step, last_step):
            render_loss_graph(loss, step)

        for param in model.parameters():
            param.data -= config.LEARNING_RATE * param.grad

        if step == first_step or step % config.LOG_EVERY_N_STEPS == 0:
            print(f"Step {step:>4} | loss {loss.data:.6f}")

    return model


def report_predictions(model: MLP) -> None:
    print("\nTarget  | Prediction")
    for inputs, target in zip(config.TRAINING_INPUTS, config.TRAINING_TARGETS, strict=True):
        print(f"{target:>7.2f} | {model(inputs).data:>10.4f}")


if __name__ == "__main__":
    trained_model = train()
    report_predictions(trained_model)
