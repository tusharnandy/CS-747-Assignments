import numpy as np
# import pulp
# import matplotlib.pyplot as plt
import argparse

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mdp', type=str, required=True)
parser.add_argument('--algorithm', type=str)
args = parser.parse_args()

algorithm = args.algorithm if args.algorithm else "vi"
file2 = open(args.mdp,"r")

data = []
for line in file2.readlines():
    data.append(line)
file2.close()

numStates = int(data[0].split(" ")[1])
numActions = int(data[1].split(" ")[1])
endStates = [int(state) for state in data[2].split(" ")[1:]]


T = np.zeros((numStates, numActions, numStates))
R = np.zeros((numStates, numActions, numStates))
i = 3
for line in data[3:]:
    info = line.split(" ")
    if info[0] == 'transition':
        s = int(info[1])
        a = int(info[2])
        s_prime = int(info[3])
        R[s, a, s_prime] = float(info[4])
        T[s, a, s_prime] = float(info[5])
        i += 1

type = data[i].split(" ")[1].split("\n")[0]
gamma = float(data[i+1].split(" ")[-1].split("\n")[0])


if type == 'episodic':
    for state in endStates:
        T[state] = np.zeros((numActions, numStates))

if algorithm == 'vi':
    # initializing variables
    V = np.zeros(numStates)
    Q = np.zeros((numStates, numActions))
    policy = np.zeros(numStates)
    theta = 1e-10
    delta = 0
    count = 0

    while count == 0 or delta > theta:
        delta = 0
        for state in range(numStates):
            v = np.round_(V[state], 9)
            for action in range(numActions):
                temp = R[state, action, :] + gamma*V[:]
                Q[state, action] = np.dot(T[state, action, :], temp)
            V[state] = np.max(Q[state, :])
            delta = max(delta, abs(v - np.round_(V[state], 9)))
        count += 1

    for state in range(numStates):
        policy[state] = np.argmax(Q[state, :])
        
        result = "{:.6f}".format(V[state]) + " " + f"{int(policy[state])}"
        print(result)

elif algorithm == "hpi":
    # Initialization
    policy = np.zeros(numStates).astype('int')
    theta = 1e-10
    delta = 1
    policy_stable = False
    count = 1

    while policy_stable is False:
        # Policy Evaluation: Matrix method
        R_pi = np.array([np.dot(T[state, policy[state], :], R[state, policy[state], :]) for state in range(numStates)])
        left_arg = np.identity(numStates) - gamma*np.array([T[state, policy[state], :] for state in range(numStates)])
        V = np.dot(np.linalg.pinv(left_arg), R_pi)


        # Policy improvement
        Q = np.zeros((numStates, numActions))
        old_policy = policy.copy()
        for state in range(numStates):
            for action in range(numActions):
                temp = R[state, action, :] + gamma*V[:]
                Q[state, action] = np.dot(T[state, action, :], temp)
            policy[state] = Q[state, :].argmax()
        if (old_policy == policy).all():
            policy_stable = True

    for state in range(numStates):
        policy[state] = np.argmax(Q[state, :])
        
        result = "{:.6f}".format(Q[state, :].max()) + " " + f"{int(policy[state])}"
        print(result)