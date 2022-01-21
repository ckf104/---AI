#include<cassert>
#include<cstdlib>
#include<ctime>
#include<iostream>
#include<sstream>
#include<fstream>
#include<algorithm>
#include<string>
#include<vector>
using namespace std;
constexpr double white = 0;
constexpr double black = 1;
constexpr double negative_max = -10000;
constexpr double positive_max = 10000;

int mycolor, comcolor;
double gridInfo[7][7]; // 先x后y，记录棋盘状态
int blackPieceCount, whitePieceCount;
static int delta[24][2] = { {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {-1, 2}, {-2, 2}, {-2, 1}, {-2, 0}, {-2, -1}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {2, -1} };
int sum_com, sum_you;//电脑和你在棋盘上的棋子数
int sum_null;//棋盘上的空地数

void menu();//菜单
void init();//初始化 
void sum();//统计棋子数目 
void print();//输出棋盘 
void gaming();//游戏进行
bool save(string);//存档 
bool load(string);//读档 
bool inMap(int, int);//判断是否越界
bool can_move(int, int);//判断（x1,y1）处的棋子是否能移动
bool can_move_all(int);//判断是否有子可走(t表示棋子的颜色，黑1白2)
bool is_digit(string& s);

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
ostream& operator<<(ostream& o, const output& con)
{
	cout << con.sx << " " << con.sy << " " << con.ex << " " << con.ey;
	return o;
}

// 向Direction方向改动坐标，并返回是否越界
inline bool MoveStep(int& x, int& y, int Direction)
{
	x = x + delta[Direction][0];
	y = y + delta[Direction][1];
	return inMap(x, y);
}

// 在坐标处落子，检查是否合法或模拟落子
bool ProcStep(int x0, int y0, int x1, int y1, double color, vector<record>* records = nullptr)
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

double evalute(double cur_color)
{
	if (cur_color == white)
		return whitePieceCount - blackPieceCount;
	else if (cur_color == black)
		return blackPieceCount - whitePieceCount;
	else
		assert(false);
}

double my_search(int depth, double cur_color, output* o = nullptr) // 返回对该player最优的结果
{
	if (depth == 0)
		return evalute(cur_color);
	vector<output> all_pos;

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
				all_pos.emplace_back(x0, y0, x1, y1);
			}
		}
	double ret = positive_max;

	for (const output& pos : all_pos)
	{
		vector<record> reds;
		int tmp_black = blackPieceCount, tmp_white = whitePieceCount;
		ProcStep(pos.sx, pos.sy, pos.ex, pos.ey, cur_color, &reds);
		auto re_value = my_search(depth - 1, 1 - cur_color);
		blackPieceCount = tmp_black, whitePieceCount = tmp_white;
		for (const record& red : reds)
		{
			gridInfo[red.x][red.y] = red.color;
		}
		if (re_value < ret)
		{
			ret = re_value;
			if (o)
				memmove(o, &pos, sizeof(output));
		}
	}
	return ret == positive_max ? evalute(cur_color) : -ret;
}

void decide(char* response, int myColor)
{
	output now_out;
	my_search(2, myColor, &now_out);
	assert(ProcStep(now_out.sx, now_out.sy, now_out.ex, now_out.ey, myColor));
	if (response)
		sprintf(response, "%d %d %d %d\n", now_out.sx, now_out.sy, now_out.ex, now_out.ey);
}

//------以上为陈克发同学的工作，以下是杨然同学的工作-------//

//判断是否越界
inline bool inMap(int x, int y) {
	if (x < 0 || x > 6 || y < 0 || y > 6)
		return false;
	else return true;
}

//输出选择菜单
void menu() {
	cout << "1:new game" << endl;
	cout << "2:load（需要额外输入加载路径，中间用空格隔开）" << endl;
	cout << "3:exit" << endl;
	int mc = 100;
	cin >> mc;
	if (mc == 1) {
		init();
		cout << "请输入数字选择先后手：" << endl;
		cout << "1:先手" << endl;
		cout << "2:后手" << endl;
		int ic;
		cin >> ic;
		while (ic > 2 || ic < 1) {
			cout << "请重新选择：" << endl;
			cin >> ic;
		}
		if (ic == 1)
		{
			mycolor = 1;
			comcolor = 0;
		}
		else if (ic == 2)
		{
			mycolor = 0;
			comcolor = 1;
			print();
			decide(nullptr, comcolor);
		}
		gaming();
	}
	else if (mc == 2) {
		string s;
		cin >> s;
		if (!load(s))
			exit(0);
		gaming();
	}
	else if (mc == 3)
		exit(0);
	else {
		cout << "error number" << endl;
		exit(0);
	}
}

int main() {
	cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
	cout << "欢迎来到同化棋小游戏，您可以输入数字进行以下操作" << endl;
	cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
	while (1) {
		menu();
	}
	return 0;
}

//初始化棋盘
void init() {
	for (int i = 0; i < 7; ++i)
		for (int j = 0; j < 7; ++j)
			gridInfo[i][j] = 0.5;
	gridInfo[0][0] = gridInfo[6][6] = 1;
	gridInfo[0][6] = gridInfo[6][0] = 0;
	blackPieceCount = 2;
	whitePieceCount = 2;
}

//存档
bool save(string path) {
	ofstream outfile;
	outfile.open(path.c_str(), ios::out | ios::trunc);
	if (!outfile) {
		cout << "存档失败，请重新输入路径" << endl;
		return false;
	}
	for (int i = 0; i < 7; ++i) {
		for (int j = 0; j < 6; ++j) {
			outfile << gridInfo[i][j] << " ";
		}
		outfile << gridInfo[i][6] << endl;
	}
	outfile << mycolor << " " << comcolor << " " << blackPieceCount << " " << whitePieceCount;
	outfile.close();
	cout << "存档成功！" << endl;
	return true;
}

//读档
bool load(string path) {
	ifstream infile;
	infile.open(path, ios::in);
	if (!infile) {
		cout << "加载失败" << endl;
		return false;
	}
	for (int i = 0; i < 7; ++i) {
		for (int j = 0; j < 7; ++j) {
			infile >> gridInfo[i][j];
		}
	}
	infile >> mycolor >> comcolor >> blackPieceCount >> whitePieceCount;
	infile.close();
	return true;
}

//判断位于（x1,y1）的棋子是否能走
bool can_move(int x1, int y1) {
	int x2, y2;
	for (int i = -2; i <= 2; ++i) {
		for (int j = -2; j <= 2; ++j) {
			if (i != 0 || j != 0) {
				x2 = x1 + i;
				y2 = y1 + j;
				if (inMap(x2, y2) && gridInfo[x2][y2] == 0.5)return 1;
			}
		}
	}
	return 0;
}

//判断某一方（颜色为t是否有子可走）
bool can_move_all(int t) {
	for (int i = 0; i < 7; ++i) {
		for (int j = 0; j < 7; ++j) {
			if (gridInfo[i][j] == t && can_move(i, j)) return 1;
		}
	}
	return 0;
}

//计数棋盘上的双方棋子数
void sum() {
	if (mycolor == 1) {
		sum_you = blackPieceCount;
		sum_com = whitePieceCount;
	}
	else {
		sum_you = whitePieceCount;
		sum_com = blackPieceCount;
	}
	sum_null = 49 - sum_you - sum_com;
}

//输出棋盘
void print() {
	sum();
	cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
	cout << "   COM:" << sum_com << "                 YOU:" << sum_you << endl;
	cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
	cout << "    0   1   2   3   4   5   6" << endl;
	cout << "  ┏ ━ ┳ ━ ┳ ━ ┳ ━ ┳ ━ ┳ ━ ┳ ━ ┓ " << endl;
	for (int i = 0; i < 6; ++i) {
		cout << " " << i << "┃";
		for (int j = 0; j < 7; ++j) {
			if (gridInfo[i][j] == 0.5)cout << "   ┃";
			else if (gridInfo[i][j] == 1)cout << " ●┃";
			else if (gridInfo[i][j] == 0)cout << " ○┃";
		}
		cout << endl << "  ┣ ━ ╋ ━ ╋ ━ ╋ ━ ╋ ━ ╋ ━ ╋ ━ ┫" << endl;
	}
	cout << " " << 6 << "┃";
	for (int j = 0; j < 7; ++j) {
		if (gridInfo[6][j] == 0.5)cout << "   ┃";
		else if (gridInfo[6][j] == 1)cout << " ●┃";
		else if (gridInfo[6][j] == 0)cout << " ○┃";
	}
	cout << endl << "  ┗ ━ ┻ ━ ┻ ━ ┻ ━ ┻ ━ ┻ ━ ┻ ━ ┛" << endl;
}

//进行游戏
void gaming() {
	if (mycolor == 1)
		cout << "您的棋子为黑棋" << endl;
	else
		cout << "您的棋子为白棋" << endl;
	int x1, y1, x2, y2;
	while (1) {
		print();
		if (!sum_null) {
			if (sum_you > sum_com)cout << "You win!" << endl;
			else cout << "You lose!" << endl;
			return;
		}
		if (!can_move_all(mycolor)) {
			cout << "You lose!" << endl;
			return;
		}
	lab:
		cout << "请输入您需要移动的棋子的行和列和移动位置，中间用空格隔开" << endl;
		string tmp1, tmp2, tmp3, tmp4;
		cin >> tmp1;
		if (tmp1 == "exit") {
			exit(0);
		}
		cin >> tmp2;
		if (tmp1 == "save")
		{
			save(tmp2);
			goto lab;
		}
		if (!is_digit(tmp1)) {
			cout << "非法输入" << endl;
			goto lab;
		}
		else {
			cin >> tmp3 >> tmp4;
			if (is_digit(tmp2) && is_digit(tmp3) && is_digit(tmp4))
				x1 = stoi(tmp1), y1 = stoi(tmp2), x2 = stoi(tmp3), y2 = stoi(tmp4);
			else {
				cout << "非法输入" << endl;
				goto lab;
			}
		}

		if (!ProcStep(x1, y1, x2, y2, mycolor)) {
			cout << "非法走子，请重试" << endl;
			goto lab;
		}
		print();
		if (!sum_null) {
			if (sum_you > sum_com)cout << "You win!" << endl;
			else cout << "You lose!" << endl;
			return;
		}

		if (!can_move_all(comcolor)) {
			cout << "You win!" << endl;
			return;
		}
		decide(nullptr, comcolor);
	}
}

bool is_digit(string& s) {
	for (int i = 0, siz = s.size(); i < siz; ++i) {
		if (s[i] >= '0' && s[i] <= '9')
			continue;
		return false;
	}
	return true;
}

