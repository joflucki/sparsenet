import torch
import sparsenet
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(20, 10)
        self.fc2 = sparsenet.nn.NormalizedLinear(10, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = sparsenet.nn.sigma(self.fc1(input))
        b = self.fc2(a)
        return b

    def param_groups(self):
        return [
            {  # This is used to indicate which parameters are used in the penalty.
                "params": [self.fc1.weight, self.fc1.bias],
                "penalized": True,
            },
            {
                # This is used to indicate which parameters are used in the penalty.
                "params": [self.fc2.weight, self.fc2.bias],
                "penalized": False,
            },
        ]


def main():
    torch.autograd.set_detect_anomaly(True)  # type: ignore

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset, weights, biases, target = sparsenet.nn.Simulation.load(
        "data/simulations/Alan"
    )

    # Neural net
    model = Net()
    model.to(device)

    # HappyLambda
    happy_lambda = sparsenet.happy_lambda.HappyLambdaMLP(
        dataset,
        100,
        [10, 1],
        device=device,
    ).run()

    trainer = sparsenet.nn.LassoMLPTrainer(
        dataset,
        target,
        model,
        [happy_lambda],
        [0.0001],
    )
    print("Running training session using LassoMLPTrainer...")
    losses = trainer.run()
    print("Final loss:", losses[-1])

    if losses:
        plt.figure(figsize=(8, 6))
        plt.plot(losses)
        plt.title("Optimization of MLP using LassoMLPTrainer")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("loss_evolution_lasso_trainer")
        plt.clf()
        print("Saved plot in loss_evolution_lasso_trainer")


if __name__ == "__main__":
    main()
