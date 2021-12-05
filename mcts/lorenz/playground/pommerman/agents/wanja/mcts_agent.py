import argparse
import multiprocessing
import inspect
import random
from gym import spaces
from queue import Empty
import numpy as np
import time
import json

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants
from pommerman import forward_model
from pommerman import characters
from pommerman import utility
from .A3C_v10_cnn_lstm import A3CNet, A3CAgent, load_checkpoint
from .sharedAdam import SharedAdam
import torch
import torch.nn.functional as F


NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18


def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)

def observe(state):
    obs_width = 5 #choose uneven number
    obs_radius = obs_width//2
    board = state['board']
    blast_strength = state['bomb_blast_strength']
    bomb_life = state['bomb_life']
    pos = np.asarray(state['position'])
    board_pad = np.pad(board,(obs_radius,obs_radius),'constant',constant_values=1)
    BS_pad = np.pad(blast_strength,(obs_radius,obs_radius),'constant',constant_values=0)
    life_pad = np.pad(bomb_life,(obs_radius,obs_radius),'constant',constant_values=0)
    #centered, padded board
    board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    bomb_BS_cent = BS_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    bomb_life_cent = life_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    ammo = np.asarray([state['ammo']])
    my_BS = np.asarray([state['blast_strength']])

    #note: on the board, 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
    out = np.empty((3,11+2*obs_radius,11+2*obs_radius),dtype=np.float32)
    out[0,:,:] = board_pad
    out[1,:,:] = BS_pad
    out[2,:,:] = life_pad
    #get raw surroundings
    raw = np.concatenate((board_cent.flatten(),bomb_BS_cent.flatten()),0)
    raw = np.concatenate((raw,bomb_life_cent.flatten()),0)
    raw = np.concatenate((raw,ammo),0)
    raw = np.concatenate((raw,my_BS),0)
    raw = np.concatenate((raw,np.zeros(6)),0)

    return out,raw

class MCTSNode(object):
    def __init__(self, p):
        self.mcts_c_puct = 1.0
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p

    def action(self):
        U = self.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N)
        return argmax_tiebreaking(self.Q + U)

    def update(self, action, reward):
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(NUM_ACTIONS)
            p[argmax_tiebreaking(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt)


class MCTSAgent(BaseAgent):
    def __init__(self, agent_id=0):
        super().__init__()
        self.agent_id = agent_id
        #self.env = self.make_env()
        self.forward_model = forward_model.ForwardModel()
        self.mcts_iters = 10
        self.temperature = 0
        self.discount = 0.99
        self.reset_tree()
        model = A3CNet()
        model, _ = load_checkpoint('A3C_v10_cnn_lstm_trained_critic_actor_1.pth', model, SharedAdam(model.parameters(), lr=0.00001))
        self.model = model
        self.hn, self.cn = self.model.get_lstm_reset()

    def init_agent(self, id_, game_type):
        self.game_type = game_type
        if inspect.isclass(self._character):
            super(MCTSAgent, self).init_agent(id_, game_type)

    def make_env(self):
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())

        return pommerman.make('PommeFFACompetition-v0', agents)

    def reset_tree(self):
        self.tree = {}

    def search(self, obs, num_iters, temperature=1):
        # remember current game state
        self.init_game_state = obs
        root = str(self._get_obs_json_info(obs))

        for i in range(num_iters):
            print(i)
            # restore game state to root node
            obs = self.init_game_state
            # serialize game state
            state = str(self._get_obs_json_info(obs))

            trace = []
            done = False
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    action = node.action()
                    trace.append((node, action))
                else: # here we reached a leaf -> use a network
                    out, raw = observe(obs)
                    logit, value, (hn, cn) = self.model(torch.from_numpy(out).float().unsqueeze(0).unsqueeze(0),
                                                        torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0),
                                                        self.hn, self.cn)

                    # OLD
                    # use unfiform distribution for probs
                    # probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                    # NEW: probs generated by network
                    probs = F.softmax(logit, dim=-1).flatten().detach().numpy()

                    # OLD
                    # use current rewards for values
                    # rewards = self.env._get_rewards()
                    # reward = rewards[self.agent_id]

                    # NEW
                    reward = value.flatten().detach().numpy()

                    # add new node to the tree

                    self.tree[state] = MCTSNode(probs)

                    # stop at leaf node
                    break

                # ensure we are not called recursively (I don't think we need this)
                # assert self.env.training_agent == self.agent_id
                # make other agents act
                print("get agents for step prediction: ")
                print(obs["board"])
                agents = self._get_agents(obs["board"], self.game_type)
                observations = _get_observations(obs)
                # obs contains observation per agent
                actions = self.forward_model.act(agents, observations, spaces.Discrete(6))
                # add my action to list of actions (where does action come from?)
                actions.insert(self.agent_id, action)
                # step environment forward
                bombs = _get_bombs(obs['bomb_blast_strength'], obs['bomb_life'], obs['bomb_moving_direction'], agents)
                # reduce number of items according to heuristic
                items = utility.make_items(obs['board'], constants.NUM_ITEMS)
                flames = _get_flames(obs['flame_life'])
                new_board, new_agents, new_bombs, new_items, new_flames = self.forward_model.step(
                actions, obs['board'], agents, bombs, items, flames)
                print("new board: ")
                print(new_board)
                new_observations = self.forward_model.get_observations(
                new_board, new_agents, new_bombs, new_flames, True, 5, self.game_type, obs['game_env'])
                print("getting observation: ")
                print(new_observations[self.agent_id])
                #obs, rewards, done, info = self.env.step(actions)
                #reward = rewards[self.agent_id]
                new_step_count = obs['step_count'] + 1
                obs = new_observations[self.agent_id]
                obs['step_count'] = new_step_count
                # fetch next state
                print("get json after simulation")
                state = str(self._get_obs_json_info(obs))

            # update tree nodes with rollout results
            for node, action in reversed(trace):
                node.update(action, reward)
                reward *= self.discount

        # reset env back where we were
        #self.env.set_json_info()
        # return action probabilities
        return self.tree[root].probs(temperature)

    def rollout(self):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        length = 0
        done = False
        while not done:
            if args.render:
                self.env.render()

            root = self.env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, args.mcts_iters, args.temperature)
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            # ensure we are not called recursively
            assert self.env.training_agent == self.agent_id
            # make other agents act
            actions = self.env.act(obs)
            # add my action to list of actions
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, info = self.env.step(actions)
            assert self == self.env._agents[self.agent_id]
            length += 1
            print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        return length, reward, rewards

    def act(self, obs, action_space):
        #obs = format_of_environment
        pi = self.search(obs, self.mcts_iters, self.temperature)
        action = np.random.choice(NUM_ACTIONS, p=pi)
        return action

    def _get_obs_json_info(self, obs):
        """Returns a json snapshot of the current game state."""
        print("get agents for json info: ")
        print(obs["board"])
        agents = self._get_agents(obs["board"], self.game_type)
        bombs = _get_bombs(obs['bomb_blast_strength'], obs['bomb_life'], obs['bomb_moving_direction'], agents)
        flames = _get_flames(obs['flame_life'])
        # reduce number of items according to heuristic
        #items = _make_items(obs['board'], constants.NUM_ITEMS, 0)
        items = {}
        ret = {
            'board_size': constants.BOARD_SIZE,
            'step_count': obs['step_count'],
            'board': obs['board'],
            'agents': agents,
            'bombs': bombs,
            'flames': flames,
            'items': [[k, i] for k, i in items.items()],
            #'intended_actions': [],
            'radio_vocab_size': constants.RADIO_VOCAB_SIZE,
            'radio_num_words': constants.RADIO_NUM_WORDS,
            # set to radio value
            '_radio_from_agent' : 0,
        }
        for key, value in ret.items():
            ret[key] = json.dumps(value, cls=utility.PommermanJSONEncoder)
        return ret

    def _get_agents(self, board, game_type):
        #init agents with game_type and id
        agents = []
        ids = {0, 1, 2, 3}
        for y in range(len(board)):
            for x in range(len(board[y])):
                if board[y][x] >= 10:
                    agent = SimpleAgent()
                    agent.init_agent(board[y][x] - 10, game_type)
                    agent._character.position = (x, y)
                    agents.append(agent)
                    print("x: " + str(x) + " y: " + str(y))
                    print(board[y][x] - 10)
                    print(ids)
                    ids.remove(board[y][x] - 10)
                    print("amk 1")
        print("amk2")
        for id in ids:
            print(str(id))
            agent = SimpleAgent()
            agent.init_agent(id, game_type)
            agent._character.die()
            agent._character.position = (-1, -1)
            agents.append(agent)
        print("amk3")
        return agents

def _make_items(board, num_items, seed = 0):
    '''Lays all of the items on the board'''
    item_positions = {}
    random.seed(seed)
    while num_items > 0:
        row = random.randint(0, len(board) - 1)
        col = random.randint(0, len(board[0]) - 1)
        if board[row, col] != constants.Item.Wood.value:
            continue
        if (row, col) in item_positions:
            continue

        item_positions[(row, col)] = random.choice([
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]).value
        num_items -= 1
    return item_positions

def _get_observations(observation):
    observations = []
    for i in range(4):
        observations.append(observation)
    return observations

def _get_bombs(blast_strength, life, moving_direction, agents):
    bombs = []
    for y in range(len(blast_strength)):
        for x in range(len(blast_strength[y])):
            if blast_strength[y][x] > 0:
                direction_greater_zero = moving_direction[y][x]
                if not direction_greater_zero:
                    direction_greater_zero = None
                # choose agent with good heuristic
                agent_id = random.randrange(4)
                agent = agents[agent_id]
                bomb = characters.Bomb(
                    agent,
                    (x, y),
                    life[y][x],
                    blast_strength[y][x].astype(int),
                    direction_greater_zero
                )
                bombs.append(bomb)
    return bombs

def _get_flames(life):
    flames = []
    for y in range(len(life)):
        for x in range(len(life)):
            if life[y][x] > 0:
                flame = characters.Flame((x, y), life[x, y])
                flames.append(flame)
    return flames

def runner(id, num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(agent_id=agent_id)

    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards = agent.rollout()
        elapsed = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, rewards, agent_id, elapsed))


def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--num_runners', type=int, default=4)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=10)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner, args=(i, args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for a new trajectory
        length, reward, rewards, agent_id, elapsed = fifo.get()

        print("Episode:", i, "Reward:", reward, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))
