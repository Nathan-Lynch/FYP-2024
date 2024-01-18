import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

class PPO():
    def __init__(self, env_name="InvertedDoublePendulum-v4", device="cuda:0"):
        self.device = device
        self.env_name = env_name
        self.input_size = 11 # HARDCODED FOR NOW
        self.num_cells = 256
        self.lr = 3e-4
        self.max_grad_norm = 1.0
        self.frame_skip = 1
        self.frames_per_batch = 1000 // self.frame_skip
        self.total_frames = 10_000 // self.frame_skip
        self.sub_batch_size = 64
        self.num_epochs = 10
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.entropy_eps = 1e-4

        # Initialize environment and components
        self.base_env = GymEnv(env_name=self.env_name, device=self.device, frame_skip=self.frame_skip)
        self.env = TransformedEnv(
            self.base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(in_keys=["observation"]),
                StepCounter(),
            ),
        )

        self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        check_env_specs(self.env)

        # Initialize neural networks
        self.actor_net = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.num_cells).to(self.device),
            nn.Tanh(),
            nn.Linear(in_features=self.num_cells, out_features=self.num_cells).to(self.device),
            nn.Tanh(),
            nn.Linear(in_features=self.num_cells, out_features=self.num_cells).to(self.device),
            nn.Tanh(),
            nn.Linear(in_features=self.num_cells * 2, out_features=self.num_cells).to(self.device),
            NormalParamExtractor(),
        )

        self.policy_module = TensorDictModule(
            self.actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.minimum,
                "max": self.env.action_spec.space.maximum,
            },
            return_log_prob=True,
        )

        self.value_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )
        
        self.value_module = ValueOperator(
            module=self.value_net,
            in_keys=["observation"],
        )

        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        self.advantage_module = GAE(
            gamma=self.gamma, lmbda=self.lmbda, value_network=self.value_module, average_gae=True
        )
        self.loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            value_target_key=self.advantage_module.value_target_key,
            critic_coef=1.0,
            gamma=0.99,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.total_frames // self.frames_per_batch)