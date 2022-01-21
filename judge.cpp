#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string.h>
#include <assert.h>

#include <WinSock2.h>
#pragma comment(lib, "ws2_32.lib")

using namespace std;
constexpr double white = 0;
constexpr double black = 1;
constexpr double negative_max = -10000;
constexpr double positive_max = 10000;

int currBotColor;      // 我所执子颜色（1为黑，0为白，棋盘状态亦同）
double gridInfo[7][7]; // 先x后y，记录棋盘状态
int blackPieceCount = 2, whitePieceCount = 2;
int startX, startY, resultX, resultY;
static int delta[24][2] = {{1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {-1, 2}, {-2, 2}, {-2, 1}, {-2, 0}, {-2, -1}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {2, -1}};

struct record
{
    int x;
    int y;
    double color;
    record(int t1, int t2, double c) : x(t1), y(t2), color(c) {}
};

struct output
{
    int sx, sy, ex, ey;
    output(int x1 = -1, int y1 = -1, int x2 = -1, int y2 = -1) : sx(x1), sy(y1), ex(x2), ey(y2) {}
};
ostream &operator<<(ostream &o, const output &con)
{
    cout << con.sx << " " << con.sy << " " << con.ex << " " << con.ey;
    return o;
}

// 判断是否在地图内
inline bool
inMap(int x, int y)
{
    if (x < 0 || x > 6 || y < 0 || y > 6)
        return false;
    return true;
}

// 向Direction方向改动坐标，并返回是否越界
inline bool MoveStep(int &x, int &y, int Direction)
{
    x = x + delta[Direction][0];
    y = y + delta[Direction][1];
    return inMap(x, y);
}

// 在坐标处落子，检查是否合法或模拟落子
bool ProcStep(int x0, int y0, int x1, int y1, double color, vector<record> *records = nullptr)
{
    if (color != 0 && color != 1)
        return false;
    if (x1 == -1 || x0 == -1) // 无路可走，跳过此回合
        return false;
    if (!inMap(x0, y0) || !inMap(x1, y1)) // 超出边界
        return false;
    if (gridInfo[x0][y0] != color)
        return false;
    int dx, dy, x, y, currCount = 0, dir;

    dx = abs((x0 - x1)), dy = abs((y0 - y1));
    if ((dx == 0 && dy == 0) || dx > 2 || dy > 2) // 保证不会移动到原来位置，而且移动始终在5×5区域内
        return false;
    if (gridInfo[x1][y1] != 0.5) // 保证移动到的位置为空
        return false;
    if (dx == 2 || dy == 2)
    { // 如果走的是5×5的外围，则不是复制粘贴
        gridInfo[x0][y0] = 0.5;
        if (records)
            records->emplace_back(x0, y0, color);
    }
    else
    {
        if (color == 1)
            blackPieceCount++;
        else
            whitePieceCount++;
    }

    gridInfo[x1][y1] = color;
    if (records)
        records->emplace_back(x1, y1, 0.5);

    for (dir = 0; dir < 8; dir++) // 影响邻近8个位置
    {
        x = x1 + delta[dir][0];
        y = y1 + delta[dir][1];
        if (!inMap(x, y))
            continue;
        if (gridInfo[x][y] == 1 - color)
        {
            if (records)
                records->emplace_back(x, y, gridInfo[x][y]);
            currCount++;
            gridInfo[x][y] = color;
        }
    }
    if (currCount != 0)
    {
        if (color == 1)
        {
            blackPieceCount += currCount;
            whitePieceCount -= currCount;
        }
        else
        {
            whitePieceCount += currCount;
            blackPieceCount -= currCount;
        }
    }
    return true;
}

bool my_search(double cur_color) // 检查该玩家是否还能移动
{
    for (int y0 = 0; y0 < 7; y0++)
        for (int x0 = 0; x0 < 7; x0++)
        {
            if (gridInfo[x0][y0] != cur_color)
                continue;
            for (int dir = 0; dir < 24; dir++)
            {
                int x1 = x0 + delta[dir][0];
                int y1 = y0 + delta[dir][1];
                if (!inMap(x1, y1))
                    continue;
                if (gridInfo[x1][y1] != 0.5)
                    continue;
                return true;
            }
        }
    return false;
}

constexpr uint16_t myport = 10000;
constexpr uint16_t black_port = 10001;
constexpr uint16_t white_port = 10002;
constexpr const char *init_str = "-1 -1 -1 -1\n";
constexpr const char *win = "-2 -2 -2 -2\n";
constexpr const char *fail = "-3 -3 -3 -3\n";
sockaddr_in sock_addr, black_addr, white_addr;
int msg_len;

SOCKET my_skt;

const char *error_msg;
char response[100];

void send_rel(const char *black_msg, const char *white_msg)
{
    sendto(my_skt, black_msg, strlen(black_msg), 0, (sockaddr *)&black_addr, sizeof(black_addr));
    sendto(my_skt, white_msg, strlen(white_msg), 0, (sockaddr *)&white_addr, sizeof(white_addr));
}

bool check_rel(double check_color)
{ // true 表示游戏结束
    auto ret = !my_search(check_color);
    if (ret)
    {
        if (check_color == white)
        {
            if (whitePieceCount <= 24)
                send_rel(win, fail);
            else
                send_rel(fail, win);
        }
        else if (check_color == black)
        {
            if (blackPieceCount <= 24)
                send_rel(fail, win);
            else
                send_rel(win, fail);
        }
        else
            assert(false);
    }
    return ret;
}

void socket_init(){
    WSADATA wsaData;
    WORD sockVersion = MAKEWORD(2,2);
    assert(WSAStartup(sockVersion, &wsaData) == 0);
}

int main()
{
    socket_init();
    my_skt = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    sock_addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    sock_addr.sin_port = htons(myport);
    sock_addr.sin_family = AF_INET;

    black_addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    white_addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    black_addr.sin_port = htons(black_port);
    white_addr.sin_port = htons(white_port);
    black_addr.sin_family = white_addr.sin_family = AF_INET;

    output o;
    if (bind(my_skt, (sockaddr *)&sock_addr, sizeof(sockaddr_in)) != 0)
    {
        cerr << "bind : ";
        cerr << WSAGetLastError() << endl;
        return -1;
    }

    while (1)
    {
        int x0, y0, x1, y1;
        // 初始化棋盘
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                gridInfo[i][j] = 0.5;
        gridInfo[0][0] = gridInfo[6][6] = 1; //|黑|白|
        gridInfo[6][0] = gridInfo[0][6] = 0; //|白|黑|
        blackPieceCount = whitePieceCount = 2;
        memmove(response, init_str, strlen(init_str) + 1);

        while (1)
        {
            if (sendto(my_skt, response, strlen(response), 0, (sockaddr *)&black_addr, sizeof(sockaddr_in)) <= 0)
            {
                error_msg = "black sendto :";
                goto err;
            }
            if ((msg_len = recvfrom(my_skt, response, 99, 0, nullptr, nullptr)) <= 0)
            {
                error_msg = "black recv : ";
                goto err;
            }
            response[msg_len] = 0;  // no default \0
            
            assert(sscanf(response, "%d %d %d %d\n", &o.sx, &o.sy, &o.ex, &o.ey) == 4);
            if(ProcStep(o.sx, o.sy, o.ex, o.ey, black, nullptr) == false){
                assert(false);
            }
            //assert(ProcStep(o.sx, o.sy, o.ex, o.ey, black, nullptr));
            if(check_rel(white))
                goto nex;
            if(sendto(my_skt, response, strlen(response), 0, (sockaddr*)&white_addr, sizeof(sockaddr_in)) <= 0){
                error_msg = "white sendto :";
                goto err;                
            }            
            if((msg_len = recvfrom(my_skt, response, 99, 0, nullptr, nullptr)) <= 0){
                error_msg = "white recv : ";
                goto err;
            }
            response[msg_len] = 0;

            assert(sscanf(response, "%d %d %d %d\n", &o.sx, &o.sy, &o.ex, &o.ey) == 4);
            assert(ProcStep(o.sx, o.sy, o.ex, o.ey, white, nullptr));
            if(check_rel(black))
                goto nex;
        }

    err:
        cout << error_msg << WSAGetLastError() << endl;
        closesocket(my_skt);
        break;
    nex:
        continue;
    }
    return 0;
}