import numpy as np

def loss(num, state):
    check = np.array([num, num, num])
    board = np.array([int(ch) for ch in state]).reshape(3,3)
    board_reflected = np.dot(board, np.array([[0,0,1],[0,1,0],[1,0,0]]))
    if (np.diag(board) == check).all() or np.diag(board_reflected == check).all():
        return True
    for r in range(3):
        if (board[r] == check).all() or (board[:, r] == check).all():
            return True
    return False

def draw(state):
    if not state.count('0'):
        return True
    return False

# collecting data for states
state_file = r"C:\Users\TUSHAR\Desktop\cs747\pa2_base\data\attt\states\states_file_p2.txt"
file1 = open(state_file,"r")
states = []
for line in file1.readlines():
    states.append(line[:-1])
file1.close()

# collecting data for policy
policy_file = r"C:\Users\TUSHAR\Desktop\cs747\pa2_base\data\attt\policies\p1_policy2.txt"
file2 = open(policy_file, 'r')
policy_data = {}
for i, line in enumerate(file2.readlines()):
    if i == 0:
        opponent = int(line[0])
        player = 3 - opponent
    else:
        line_data = line.split('\n')[0].split(' ')
        policy_data[line_data[0]] = [float(p) for p in line_data[1:]]
file2.close()


transition_data = []
end_states = []

for state_index, state in enumerate(states):                        # selecting a valid state
    for action in [i for i, x in enumerate(state) if x == "0"]:     # for all positions of '0'
        action_outcome = state[:action] + '2' + state[action+1:]    # outcome of action = filled_cell
        if draw(action_outcome) or loss(player, action_outcome):    # if our player's move ends with 0 reward
            if action_outcome not in end_states:                    # if action_outcome ended for first time
                end_states.append(action_outcome)                   # add it as an end state
            transition_data.append(f"transition {state_index} {action} {len(states)+end_states.index(action_outcome)} 0 1")
        else:
            for i, prob in enumerate(policy_data[action_outcome]):
                if prob != 0.0:
                    next_state = action_outcome[:i] + '1' + action_outcome[i+1:]
                    if loss(opponent, next_state):
                        if next_state not in end_states:
                            end_states.append(next_state)
                        transition_data.append(f"transition {state_index} {action} {len(states)+end_states.index(next_state)} 1 {prob}")
                    elif draw(next_state):
                        if next_state not in end_states:
                            end_states.append(next_state)
                        transition_data.append(f"transition {state_index} {action} {len(states)+end_states.index(next_state)} 0 {prob}")
                    else:
                        transition_data.append(f"transition {state_index} {action} {states.index(next_state)} 0 {prob}")

new = len(end_states)
print(f"Encoded {new} states as end states")
for i in range(10):
    print(transition_data[i])

text = [f"numStates {len(states)+len(end_states)}",
        f"numActions {9}"]
end_line = 'end'
for i, es in enumerate(end_states):
    end_line += f' {len(states)+i}'        
text.append(end_line)

for line in transition_data:
    text.append(line)

text.append("mdptype episodic\ndiscount 0.9")

file3 = open(r"C:\Users\TUSHAR\Desktop\cs747\pa2_base\data\attt\mdp\sample_mdp.txt", 'w')
for line in text:
    print(line)
    file3.write(line)
    file3.write('\n')
file3.close()