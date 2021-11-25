import random
from gym import spaces
from . import BaseAgent
from . import SimpleAgent
from .. import constants
from .. import utility
from .. import forward_model
from .. import characters

class OurAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(OurAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        print(obs)
        model = forward_model.ForwardModel()

        agents = _get_agents(obs["board"], self.game_type)
        observations = _get_observations(obs)
        # obs contains observation per agent
        actions = model.act(agents, observations, action_space)

        bombs = _get_bombs(obs['bomb_blast_strength'], obs['bomb_life'], obs['bomb_moving_direction'], agents)
        # reduce number of items according to heuristic
        items = utility.make_items(obs['board'], constants.NUM_ITEMS)
        flames = _get_flames(obs['flame_life'])
        new_board, new_agents, new_bombs, new_items, new_flames = model.step(
        actions, obs['board'], agents, bombs, items, flames)
        #print("new board:")
        #print(new_board)
        #print("new agents:")
        #print(new_agents)
        #print("new bombs:")
        #print(new_bombs)
        #print("new items:")
        #print(new_items)
        #print("new flames:")
        #print(new_flames)
        #new_observations = model.get_observations(new_board, new_agents,
        #new_bombs, new_flames)
        return action_space.sample()

    def init_agent(self, id_, game_type):
        self.game_type = game_type
        super(OurAgent, self).init_agent(id_, game_type)

def _get_observations(observation):
    observations = []
    for i in range(4):
        observations.append(observation)
    return observations

def _get_agents(board, game_type):
    #print(observation)
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
                ids.remove(board[y][x] - 10)
    for id in ids:
        agent = SimpleAgent()
        agent.init_agent(id, game_type)
        agent._character.die()
        agents.append(agent)
    return agents

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
