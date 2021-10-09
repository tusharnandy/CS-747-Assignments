import numpy as np

def loss(num, state):
    check = np.array([num, num, num])
    board = np.array([int(ch) for ch in state]).reshape(3,3)
    print(board)
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

if __name__ == "__main__":
    print(loss(2, '121221120'))