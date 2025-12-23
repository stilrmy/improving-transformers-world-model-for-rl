import pytest
import torch

from improving_transformers_world_model import (
    WorldModel,
    Agent,
    Impala
)

from improving_transformers_world_model.mock_env import Env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.mark.parametrize('critic_use_regression', (False, True))
@pytest.mark.parametrize('actor_use_world_model_embed', (False, True))
@pytest.mark.parametrize('continuous_action_range', (0.5, (-1., 2.)))
@pytest.mark.parametrize('use_continuous_actions', (False, True))
def test_agent(
    critic_use_regression,
    actor_use_world_model_embed,
    continuous_action_range,
    use_continuous_actions
):

    # world model

    if use_continuous_actions:
        action_kwargs = dict(num_actions = 0, action_dim = 2)
    else:
        action_kwargs = dict(num_actions = 5)

    image_size = 35

    world_model = WorldModel(
        image_size = image_size,
        patch_size = 7,
        channels = 3,
        reward_num_bins = 10,
        **action_kwargs,
        transformer = dict(
            dim = 32,
            depth = 1,
            block_size = 25
        ),
        tokenizer = dict(
            dim = 7 * 7 * 3,
            distance_threshold = 0.5
        )
    ).to(device)

    state = torch.randn(2, 3, 8, image_size, image_size)
    rewards = torch.randint(0, 10, (2, 8)).float()
    if use_continuous_actions:
        actions = torch.randn(2, 8, action_kwargs.get('action_dim'))
    else:
        actions = torch.randint(0, 5, (2, 8, 1))
    is_terminal = torch.randint(0, 2, (2, 8)).bool()

    loss = world_model(state, actions = actions, rewards = rewards, is_terminal = is_terminal)
    loss.backward()

    # agent

    if not use_continuous_actions and isinstance(continuous_action_range, tuple):
        pytest.skip('continuous action ranges only apply to continuous actions')

    action_range = continuous_action_range if use_continuous_actions else 1.

    agent = Agent(
        impala = dict(
            image_size = image_size,
            channels = 3
        ),
        actor = dict(
            dim = 32,
            num_actions = None if use_continuous_actions else 5,
            action_dim = action_kwargs.get('action_dim') if use_continuous_actions else None,
            dim_world_model_embed = 32 if actor_use_world_model_embed else None
        ),
        critic = dict(
            dim = 64,
            use_regression = critic_use_regression
        ),
        action_range = action_range
    ).to(device)

    if use_continuous_actions:
        dummy_frame = torch.zeros(1, 3, image_size, image_size, device = device)
        actor_input, _ = agent.impala(dummy_frame)
        dist = agent.actor.get_dist(actor_input)

        if isinstance(continuous_action_range, tuple):
            low, high = continuous_action_range
            expected_scale = (high - low) / 2
            expected_offset = (high + low) / 2
        else:
            expected_scale = continuous_action_range
            expected_offset = 0.

        assert torch.allclose(agent.actor.action_scale, torch.full_like(agent.actor.action_scale, expected_scale))
        assert torch.allclose(agent.actor.action_offset, torch.full_like(agent.actor.action_offset, expected_offset))
        assert torch.allclose(dist.stddev, torch.full_like(dist.stddev, expected_scale))

    env = Env((3, image_size, image_size))

    dream_memories = agent(
        world_model,
        state[0, :, 0],
        max_steps = 5,
        use_world_model_embed = actor_use_world_model_embed
    )

    real_memories = agent.interact_with_env(
        env,
        world_model = world_model if actor_use_world_model_embed else None,
        max_steps = 5
    )

    agent.learn([dream_memories, real_memories])

# burn-in

def world_model_burn_in():
    world_model = WorldModel(
        image_size = 63,
        patch_size = 7,
        channels = 3,
        reward_num_bins = 10,
        num_actions = 5,
        transformer = dict(
            dim = 32,
            depth = 1,
            block_size = 81
        ),
        tokenizer = dict(
            dim = 7 * 7 * 3,
            distance_threshold = 0.5
        )
    ).to(device)

    state = torch.randn(2, 3, 20, 63, 63) # batch, channels, time, height, width - craftax is 3 channels 63x63, and they used rollout of 20 frames. block size is presumably each image
    rewards = torch.randint(0, 10, (2, 20)).float()
    actions = torch.randint(0, 5, (2, 20, 1))
    is_terminal = torch.randint(0, 2, (2, 20)).bool()

    loss = world_model(state, actions = actions, rewards = rewards, is_terminal = is_terminal)
    loss.backward()

    _, burn_in_cache = world_model(
        state,
        actions = actions,
        rewards = rewards,
        is_terminal = is_terminal,
        return_cache = True,
        return_loss = False,
        detach_cache = True,
    )

    loss = world_model(
        state,
        actions = actions,
        rewards = rewards,
        is_terminal = is_terminal,
        return_loss = True,
        cache = burn_in_cache,
        remove_cache_len_from_time = False
    )

    loss.backward()

def impala_burn_in():
    impala = Impala()
    images = torch.randn(5, 3, 63, 63)

    _, gru_hidden = impala(images)
    cnn_out_rnn_out, gru_hidden = impala(images, gru_hidden = gru_hidden.detach())
