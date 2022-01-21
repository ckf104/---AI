import sys
import random
import torch
import socket
from torch import nn

# 1 表示黑色棋子，-1 表示白色棋子
gridInfo = [[0 for _ in range(7)] for _ in range(7)]
gridInfo[0][0] = gridInfo[6][6] = 1
gridInfo[0][6] = gridInfo[6][0] = -1
delta = [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2],
         [-1, 2], [-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1]]
piece_count = [0, 2, 2]  # piece_count[-1] = white, 1 = black
my_color = -1


class evalNet(nn.Module):
    def __init__(self):
        super(evalNet, self).__init__()
        self.l1 = nn.Linear(49, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, 1)
        # nn.init.uniform_(self.l1.weight, -1, 1)
        # nn.init.uniform_(self.l2.weight, -1, 1)
        # nn.init.uniform_(self.l3.weight, -1, 1)
        # nn.init.uniform_(self.l4.weight, -1, 1)
        self.linear_net = nn.Sequential(
            self.l1,
            nn.LeakyReLU(0.5),
            self.l2,
            nn.LeakyReLU(0.5),
            self.l3,
            nn.LeakyReLU(0.5),
            self.l4,
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear_net(x)


class Bellman_loss(nn.Module):
    def __init__(self):
        super(Bellman_loss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(torch.sub(x, y), 2))


discount = 0.5
espison = 5
batch = 4
device = torch.device("cpu")
a = evalNet()
a.to(device)

start_point = 90000

a.load_state_dict(torch.load(f"data/state_{sys.argv[1]}_{start_point}.pth"))
bellman_loss = Bellman_loss()
opt = torch.optim.Adam(a.parameters(), lr=5e-4)
experience_pool = []  # last_state, reward, now_state, last_state是落子后的局面，last_piece_count是该轮落子前的状态
last_state = torch.tensor([])
last_piece_count = [i for i in piece_count]
first = 1


# torch.save(a.state_dict(), "./state.pth")

def getTensor(cur_color):
    tmp = []
    for i in gridInfo:
        for j in i:
            tmp.append(j * cur_color)
    return torch.tensor(tmp, dtype=torch.float32).to(device)


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


my_skt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
my_port = 0
if sys.argv[1] == "white":
    my_port = 10002
elif sys.argv[1] == "black":
    my_port = 10001
else:
    print("error color")
    exit(-1)

counter = 0
my_skt.bind(("127.0.0.1", my_port))


def save():
    if (counter % 10000) == 0:
        torch.save(a.state_dict(), f"./data/state_{sys.argv[1]}_{counter + start_point}.pth")


@torch.no_grad()
def predict_value(res: list):
    reds = []
    black_count, white_count = piece_count[1], piece_count[-1]
    procStep(res[0], res[1], res[2], res[3], my_color, reds)
    val = a(getTensor(my_color))
    piece_count[1], piece_count[-1] = black_count, white_count
    for x, y, col in reds:
        gridInfo[x][y] = col
    return val


def find_best(all_pos: list):
    max_value = -10000
    respon = []
    for res in all_pos:
        val = predict_value(res)
        if val > max_value:
            max_value = val
            respon = res

    return max_value, respon


def update(last_tensor, reward, now_value, is_terminate=0):  # all parameter are tensor
    opt.zero_grad()
    last_value = a(last_tensor)
    reward = reward.to(device)
    if is_terminate:
        miss = bellman_loss(last_value, reward)
    else:
        miss = bellman_loss(last_value, reward + discount * now_value)
    miss.backward()
    opt.step()


def batch_update():
    t = len(experience_pool)
    for i in range(t):
        train_batch = [random.choice(experience_pool) for _ in range(batch)]
        reward = torch.tensor([i[1] for i in train_batch])
        last = torch.stack([i[0] for i in train_batch])
        max_value = torch.tensor([i[2] for i in train_batch])
        update(last, reward, max_value)


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

    global last_state
    global last_piece_count

    ran = random.randint(0, 100)
    if ran <= espison:
        response = random.choice(all_pos)
        max_value = predict_value(response)
    else:
        max_value, response = find_best(all_pos)

    if max_value != -10000:  # 先记录落子前的piece_count，再进行模拟下一步，再获取当前状态
        global first
        if first:
            first = 0
            last_piece_count = [i for i in piece_count]
            procStep(response[0], response[1], response[2], response[3], my_color, [])
            last_state = getTensor(my_color)
        else:
            reward = torch.tensor((piece_count[my_color] - last_piece_count[my_color]) / 5)
            experience_pool.append([last_state, reward, max_value])
            # batch_update()
            # update(last_state, reward, max_value)
            last_piece_count = [i for i in piece_count]
            procStep(response[0], response[1], response[2], response[3], my_color, [])
            last_state = getTensor(my_color)

    msg = f"{response[0]} {response[1]} {response[2]} {response[3]}\n"
    my_skt.sendto(bytes(msg, encoding="utf-8"), judge)


while 1:
    piece_count = [0, 2, 2]
    my_color = -1
    gridInfo = [[0 for _ in range(7)] for _ in range(7)]
    gridInfo[0][0] = gridInfo[6][6] = 1
    gridInfo[0][6] = gridInfo[6][0] = -1
    first = 1
    experience_pool = []

    while 1:
        response, judge = my_skt.recvfrom(100)
        response = bytes.decode(response)
        response = response[:len(response) - 1]
        x0, y0, x1, y1 = [int(i) for i in response.split()]
        if x0 == -2:
            counter += 1
            experience_pool.append([last_state, torch.tensor(5), torch.tensor(0)])
            # update(last_state, torch.tensor(5, device=device), torch.tensor(0, device=device), 1)
            batch_update()
            print(counter, "my win")
            save()
            break
        elif x0 == -3:
            counter += 1
            experience_pool.append([last_state, torch.tensor(-5), torch.tensor(0)])
            batch_update()
            # update(last_state, torch.tensor(-5, device=device), torch.tensor(0, device=device), 1)
            print(counter, "my fail")
            save()
            break
        elif x0 == -1:
            my_color = 1
        else:
            procStep(x0, y0, x1, y1, -my_color, [])
        decide()
