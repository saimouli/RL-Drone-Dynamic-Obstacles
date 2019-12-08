import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        print("Runner is initiated")

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []

        # For n in range number of steps
        self.obs[:] = self.env.reset()
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            print("The action is {} //////////////".format(actions))
            print("Done status just after running the step {}".format(self.dones))
            

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            print("Reward is {}".format(rewards))
            print("Done robots are {}".format(self.dones))
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            print("debugging mb_rewards after append {}".format(mb_rewards))
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards) # 8,1
        print("debugging mb_return shape {}".format(mb_returns.shape))
        mb_advs = np.zeros_like(mb_rewards) #8,1
        print("debugging mb_adv shape {}".format(mb_rewards.shape))
        lastgaelam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        print("debugging mb_values after append {}".format(mb_values))
        print("debugging mb_advs after append {}".format(mb_advs))
        # print("mb_values is {}".format(np.shape(mb_values)[0],1))
        # print("mb_advs is {}".format(np.shape(mb_advs)[0],1))
        mb_values.shape = (np.shape(mb_values)[0],1)
        mb_advs.shape = (np.shape(mb_advs)[0],1)
        mb_returns = mb_advs + mb_values
        print("The output sum shape {}".format(mb_returns.shape))

        # Added by Sai
        mb_dones.shape = (np.shape(mb_dones)[0],1)

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
        
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    # print("-------------------------------")
    # print("The array in s {}".format(arr))
    # print("Type of the array in sf01 {}".format(type(arr)))
    # print("shape in sf01 {}".format(s))
    # print("The return array after the operation {}".format(arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]))) #( 8 * 2 ,)
    # print("After reshape {}".format(arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]).shape))
    # print("--------------------------------")
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


