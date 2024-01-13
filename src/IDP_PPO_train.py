from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm
from PPO import PPO

model = PPO()

logs = defaultdict(list)
pbar = tqdm(total=model.total_frames * model.frame_skip)
eval_str = ""

for i, tensordict_data in enumerate(model.collector):
    with torch.no_grad():
        model.advantage_module(tensordict_data)

    for _ in range(model.num_epochs):
        data_view = tensordict_data.reshape(-1)
        model.replay_buffer.extend(data_view.cpu())

        for _ in range(model.frames_per_batch // model.sub_batch_size):
            model.advantage_module(tensordict_data)
            subdata = model.replay_buffer.sample(model.sub_batch_size)
            loss_vals = model.loss_module(subdata.to(model.device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.loss_module.parameters(), model.max_grad_norm)
            model.optim.step()
            model.optim.zero_grad()

    
    
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel() * model.frame_skip)
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(model.optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            eval_rollout = model.env.rollout(1000, model.policy_module)

            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )

            logs["eval step_count"].append(eval_rollout["step_count"].max().item())

            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout

    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    model.scheduler.step() #Learning rate scheduler

torch.save(model.policy_module.state_dict(), 'C:/Users/natha/Desktop/FYP-2024/src/trained_models/trained_IDP_PPO_model.pth')

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()