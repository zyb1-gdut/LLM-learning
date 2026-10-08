import pygame
import sys
import random
import numpy as np

# 初始化pygame
pygame.init()

# 游戏常量
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 8  # 8x8的网格
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, ORANGE]

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("消消乐小游戏")
clock = pygame.time.Clock()

# 游戏字体
font = pygame.font.SysFont(None, 36)

class Game:
    def __init__(self):
        self.grid = []
        self.selected_cell = None
        self.score = 0
        self.generate_grid()

    def generate_grid(self):
        """生成初始网格"""
        self.grid = [[random.choice(COLORS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        # 确保初始状态没有可消除的组合
        while self.check_matches():
            self.grid = [[random.choice(COLORS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    def draw(self):
        """绘制游戏界面"""
        screen.fill(WHITE)

        # 绘制网格
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = self.grid[row][col]
                rect = pygame.Rect(col * GRID_WIDTH, row * GRID_HEIGHT, GRID_WIDTH, GRID_HEIGHT)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)  # 边框

                # 如果被选中，绘制高亮边框
                if self.selected_cell == (row, col):
                    pygame.draw.rect(screen, WHITE, rect, 5)

        # 显示分数
        score_text = font.render(f"分数: {self.score}", True, BLACK)
        screen.blit(score_text, (10, HEIGHT - 40))

        # 显示操作提示
        hint_text = font.render("点击两个相邻的方块进行交换", True, BLACK)
        screen.blit(hint_text, (WIDTH // 2 - hint_text.get_width() // 2, 10))

    def get_cell(self, pos):
        """获取鼠标位置对应的网格坐标"""
        x, y = pos
        return y // GRID_HEIGHT, x // GRID_WIDTH

    def check_matches(self):
        """检查是否有匹配的方块"""
        matches = []

        # 检查水平匹配
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - 2):
                if self.grid[row][col] == self.grid[row][col + 1] == self.grid[row][col + 2]:
                    matches.append((row, col))
                    matches.append((row, col + 1))
                    matches.append((row, col + 2))

        # 检查垂直匹配
        for col in range(GRID_SIZE):
            for row in range(GRID_SIZE - 2):
                if self.grid[row][col] == self.grid[row + 1][col] == self.grid[row + 2][col]:
                    matches.append((row, col))
                    matches.append((row + 1, col))
                    matches.append((row + 2, col))

        return list(set(matches))  # 去除重复项

    def remove_matches(self, matches):
        """移除匹配的方块并增加分数"""
        for row, col in matches:
            self.grid[row][col] = None
            self.score += 10

    def fall_blocks(self):
        """让方块下落填补空缺"""
        # 从底部向上处理每一列
        for col in range(GRID_SIZE):
            empty_spaces = 0
            # 从底部向上遍历
            for row in range(GRID_SIZE - 1, -1, -1):
                if self.grid[row][col] is None:
                    empty_spaces += 1
                elif empty_spaces > 0:
                    # 移动当前方块到空位
                    self.grid[row + empty_spaces][col] = self.grid[row][col]
                    self.grid[row][col] = None

        # 在顶部生成新方块
        for col in range(GRID_SIZE):
            for row in range(GRID_SIZE):
                if self.grid[row][col] is None:
                    self.grid[row][col] = random.choice(COLORS)

    def swap_cells(self, cell1, cell2):
        """交换两个单元格的内容"""
        row1, col1 = cell1
        row2, col2 = cell2
        self.grid[row1][col1], self.grid[row2][col2] = self.grid[row2][col2], self.grid[row1][col1]

    def is_adjacent(self, cell1, cell2):
        """检查两个单元格是否相邻"""
        row1, col1 = cell1
        row2, col2 = cell2
        return (abs(row1 - row2) == 1 and col1 == col2) or (abs(col1 - col2) == 1 and row1 == row2)

    def update(self):
        """更新游戏状态"""
        matches = self.check_matches()
        if matches:
            self.remove_matches(matches)
            self.fall_blocks()
            # 递归检查是否有新的匹配
            new_matches = self.check_matches()
            while new_matches:
                self.remove_matches(new_matches)
                self.fall_blocks()
                new_matches = self.check_matches()

# 创建游戏实例
game = Game()

# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            cell = game.get_cell(pos)

            if game.selected_cell is None:
                game.selected_cell = cell
            elif game.selected_cell == cell:
                game.selected_cell = None
            elif game.is_adjacent(game.selected_cell, cell):
                # 交换方块
                game.swap_cells(game.selected_cell, cell)
                game.selected_cell = None
                # 更新游戏状态
                game.update()

    # 绘制游戏
    game.draw()
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
