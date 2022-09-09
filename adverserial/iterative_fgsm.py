import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import AverageMeter


class IterativeFGSM:
    def __init__(
        self,
        num_iterations,
        epsilon,
        loss=F.nll_loss,
        verbose=False,
    ) -> None:
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.alpha = self.epsilon / self.num_iterations

        self.loss = loss

        self.verbose = verbose

    def attack(self, model, data, target, device=torch.device("cpu")):
        """
            Attack
        """

        target = torch.tensor(target)
        # target = F.one_hot(target, num_classes=10)
        data, target = data.to(device), target.to(device)

        if len(data.shape)!=4:
            data = data.unsqueeze(0)
        if len(target.shape)!=1:
            target = target.unsqueeze(0)

        perturbed = data.clone()
        if self.alpha==0:
            return perturbed

        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            if self.verbose:
                print("It is already predicted wrong.")
            return data
            
        _loss = self.loss(output, target)

        model.zero_grad()
        _loss.backward()
        data_grad = data.grad.data

        

        for _ in range(self.num_iterations):
            perturbed = perturbed + self.alpha * data_grad.sign()
            perturbed = torch.clamp(perturbed, 0, 1)
            if torch.norm((perturbed-data), p=float('inf')) > self.epsilon:
                break

        return perturbed

    @staticmethod
    def test(
        iterative_fgsm,
        model,
        dataset,
        device
    ):
        top1 = AverageMeter()

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )

        report = pd.DataFrame(
            {
                "sample_id": list(),
                "output": list(),
                "perturbed_output": list(),
                "correctness": list()
            }
        )

        with tqdm(data_loader) as t_data_loader:
            for it, (data, target) in enumerate(t_data_loader):
                t_data_loader.set_description(
                    "Iterative FGSM ( epsilone = {:.5e} )".format(iterative_fgsm.alpha))

                # target = target.unsqueeze(0)


                perturbed = iterative_fgsm.attack(
                    model=model,
                    data=data,
                    target=target,
                    device=device
                )

                output = model(perturbed)
                final_pred = output.max(1, keepdim=True)[1]

                correctness = final_pred.item() == target.item()
                correctness = 1 if correctness else 0

                top1.update(correctness)

                t_data_loader.set_postfix(
                    acc="{:.3f}".format(top1.avg),
                )

                report = report.append({
                    "epsilon": iterative_fgsm.alpha,
                    "sample_id": it,
                    "output": target.cpu().detach().numpy().tolist()[0],
                    "perturbed_output": final_pred.item(),
                    "correctness": correctness
                },
                    ignore_index=True
                )

        return report, top1
