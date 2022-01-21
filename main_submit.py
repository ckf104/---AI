import sys
import torch
from torch import nn

# 1 表示黑色棋子，-1 表示白色棋子
gridInfo = [[0 for _ in range(7)] for _ in range(7)]
gridInfo[0][0] = gridInfo[6][6] = 1
gridInfo[0][6] = gridInfo[6][0] = -1
delta = [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2],
         [-1, 2], [-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1]]
piece_count = [0, 2, 2]  # piece_count[-1] = white, 1 = black
my_color = -1


def evaluate_1(cur_color):
    return piece_count[cur_color] - piece_count[-cur_color]


def recover(record, white, black):
    for x, y, color in record:
        gridInfo[x][y] = color
    piece_count[1] = black
    piece_count[-1] = white


class evalNet(nn.Module):
    def __init__(self):
        super(evalNet, self).__init__()
        self.l1 = nn.Linear(49, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, 1)
        self.linear_net = nn.Sequential(
            self.l1,
            nn.LeakyReLU(0.5),
            self.l2,
            nn.LeakyReLU(0.5),
            self.l3,
            nn.LeakyReLU(0.5),
            self.l4,
            nn.LeakyReLU(0.5),
        )

    def forward(self, x):
        return self.linear_net(x)


net = [0, evalNet(), evalNet()]
net[1].load_state_dict(torch.load("/data/state_black_150000.pth"))
net[-1].load_state_dict(torch.load("/data/state_white_150000.pth"))


def getTensor(cur_color):
    tmp = []
    for i in gridInfo:
        for j in i:
            tmp.append(j * cur_color)
    return torch.tensor(tmp, dtype=torch.float32)


def inMap(x, y):
    if x < 0 or x > 6 or y < 0 or y > 6:
        return False
    return True


def procStep(sx, sy, ex, ey, color, records: list):
    if (not inMap(sx, sy)) or (not inMap(ex, ey)):
        return False
    if sx == -1 or ex == -1:
        return False
    if gridInfo[sx][sy] != color or gridInfo[ex][ey] != 0:
        return False

    dx = abs(sx - ex)
    dy = abs(sy - ey)
    if dx > 2 or dy > 2 or (dx == 0 and dy == 0):
        return False
    if dx == 2 or dy == 2:
        gridInfo[sx][sy] = 0
        records.append([sx, sy, color])
    else:
        piece_count[color] += 1
    gridInfo[ex][ey] = color
    records.append([ex, ey, 0])

    for i in range(8):
        x, y = ex + delta[i][0], ey + delta[i][1]
        if not inMap(x, y):
            continue
        if gridInfo[x][y] == -color:
            records.append([x, y, -color])
            gridInfo[x][y] = color
            piece_count[color] += 1
            piece_count[-color] -= 1


# 估值函数用来评估该对方走时的局面情况，值越大对己方越有利。选择搜索两步，自己走一步后对方再走一步，
# 先用第一个估值函数判断此时局面，如果相同时用第二个估值函数判断，因此在min_decide中，应该对方选择
# 己方估值函数1较大，相同时估值函数2较大的局面，
# 表示对己方越有利。decide函数需要返回对己方最有利的决策

def find_all_pos(cur_color):
    all_pos = []
    for x in range(7):
        for y in range(7):
            if gridInfo[x][y] != cur_color:
                continue
            for dx, dy in delta:
                nx, ny = x + dx, y + dy
                if (not inMap(nx, ny)) or gridInfo[nx][ny] != 0:
                    continue
                all_pos.append([x, y, nx, ny])
    return all_pos


# 上层的最大节点表示，经过前面分支的搜索，至少可以拿到 min_value_1, min_value_2这样的结果（最小节点的返回）
# 因此选择另一个分支，发现min_decide可以返回比这个结果更大的值，那么就没必要探索这边了，即该函数返回false
def min_decide(cur_color, depth, my_can_min_value_1, my_can_min_value_2):
    if depth == 1:
        all_pos = find_all_pos(cur_color)
        now_can_max_value_1, now_can_max_value_2 = -10000, -10000
        for pos in all_pos:
            record = []
            white_count, black_count = piece_count[-1], piece_count[1]
            procStep(pos[0], pos[1], pos[2], pos[3], cur_color, record)
            value_1 = evaluate_1(cur_color)
            if value_1 > my_can_min_value_1:
                recover(record, white_count, black_count)
                return [False, 0, 0]
            if value_1 < now_can_max_value_1:
                recover(record, white_count, black_count)
                continue
            value_2 = net[cur_color](getTensor(cur_color))
            if value_1 == my_can_min_value_1 and value_2 > my_can_min_value_2:
                recover(record, white_count, black_count)
                return [False, 0, 0]
            if value_1 > now_can_max_value_1 or value_2 > now_can_max_value_2:
                now_can_max_value_1, now_can_max_value_2 = value_1, value_2
            recover(record, white_count, black_count)
        return [True, now_can_max_value_1, now_can_max_value_2]
    else:
        assert False


# 上层的最小节点表示，经过前面分支的搜索，至少可以拿到 max_value_1, max_value_2 这样的结果
# 在满足value_1, value_2 > max_value_1, max_value_2的前提下，尽可能把返回值压小
def max_decide(cur_color, depth, my_can_max_value_1, my_can_max_value_2):
    if depth == 1:
        assert False
    else:
        all_pos = find_all_pos(cur_color)
        now_can_min_value_1, now_can_min_value_2 = 10000, 10000
        proc = []
        for pos in all_pos:
            record = []
            white_count, black_count = piece_count[-1], piece_count[1]
            procStep(pos[0], pos[1], pos[2], pos[3], cur_color, record)
            rel, value_1, value_2 = min_decide(-cur_color, depth - 1, now_can_min_value_1, now_can_min_value_2)
            if not rel:
                recover(record, white_count, black_count)
                continue
            if (value_1, value_2) <= (my_can_max_value_1, my_can_max_value_2):
                recover(record, white_count, black_count)
                return False, proc
            elif (value_1, value_2) < (now_can_min_value_1, now_can_min_value_2):
                now_can_min_value_1, now_can_min_value_2 = value_1, value_2
                proc = pos
            recover(record, white_count, black_count)
        return True, proc


def decide():
    bo, response = max_decide(my_color, 2, -100000, -100000)
    procStep(response[0], response[1], response[2], response[3], my_color, [])
    print(response[1], response[0], response[3], response[2])
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
    # msg = f"{response[0]} {response[1]} {response[2]} {response[3]}\n"
    # my_skt.sendto(bytes(msg, encoding="utf-8"), ("127.0.0.1", 10000))


"""
def decide():
    all_pos = []
    for x in range(7):
        for y in range(7):
            if gridInfo[x][y] != my_color:
                continue
            for dx, dy in delta:
                nx, ny = x + dx, y + dy
                if (not inMap(nx, ny)) or gridInfo[nx][ny] != 0:
                    continue
                all_pos.append([x, y, nx, ny])

    max_value = -10000
    response = []
    for res in all_pos:
        reds = []
        black_count, white_count = piece_count[1], piece_count[-1]
        procStep(res[0], res[1], res[2], res[3], my_color, reds)
        val = a(getTensor(my_color))
        if val > max_value:
            max_value = val
            response = res
        piece_count[1], piece_count[-1] = black_count, white_count
        for x, y, col in reds:
            gridInfo[x][y] = col
    if max_value != -10000:
        procStep(response[0], response[1], response[2], response[3], my_color, [])
    print(response[1], response[0], response[3], response[2])
    print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
"""

first = 1
while 1:
    if first:  # 上交时需要打开这三行的注释，并切换模型的加载路径，并注释掉socket相关代码, x, y也记得交换
        first = 0
        input()
    s = ""
    while not len(s):
        s = input()
        # t = bytes.decode(my_skt.recvfrom(100)[0])
        # s = t[:len(t) - 1]

    y0, x0, y1, x1 = [int(i) for i in s.split()]
    # x0, y0, x1, y1 = [int(i) for i in s.split()]
    if x0 == -1:
        my_color = 1
    else:
        procStep(x0, y0, x1, y1, -my_color, [])

    decide()
    sys.stdout.flush()
