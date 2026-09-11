""" 测试环境(随机移动) """
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core import make_env, HouseEnv, name_transform, DoorColor, RoomColor
from common.project_paths import CONFIG_PATH


class InteractiveHouseEnv(object):
    """ 房屋交互环境 """
    allow_modes = ["Increase", "Decrease"]

    def __init__(self, env:HouseEnv) -> None:
        self.env = env
        self.selected_room_idx = None
        self.mode = "Increase"
        self.show_room_names = True
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """ 设置回调函数 """
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event):
        """ 鼠标点击事件 """
        if event.inaxes != self.ax:
            return
            
        # 检查点击是否在某个房间内
        for i, room in enumerate(self.env.rooms):
            if (room.x1 <= event.xdata <= room.x1 + room.width and 
                room.y1 <= event.ydata <= room.y1 + room.height):
                self.selected_room_idx = i
                print(f"选中房间: {room.room_type.name}")
                self.render()
                return
                
        # 如果点击在房间外，取消选择
        self.selected_room_idx = None
        self.render()

    def on_key_press(self, event):
        """ 键盘按键事件 """
        if self.selected_room_idx is None:
            print("请先选择一个房间")
            return
        
        room = self.env.rooms[self.selected_room_idx]
        action = [4] * len(self.env.rooms)  # 所有房间默认不动
        # 映射按键到动作
        key_actions = {
            'up': 0,    # 上移
            'down': 1,  # 下移
            'left': 2,  # 左移
            'right': 3, # 右移
            'y': 5,     # 增加/减少上边高度
            'h': 7,     # 增加/减少下边高度
            'g': 9,     # 增加/减少左边宽度
            'j': 11,     # 增加/减少右边宽度
        }

        if event.key in key_actions:
            if event.key in ['up', 'down', 'left', 'right']:
                action[self.selected_room_idx] = key_actions[event.key]
            elif event.key in ['y', 'h', 'g', 'j']:
                if self.mode == "Increase":
                    action[self.selected_room_idx] = key_actions[event.key]
                else:
                    action[self.selected_room_idx] = key_actions[event.key] + 1
            state, reward, done, info = self.env.step(action)
            print("当前房间`{}`, 动作: `{}`, 奖励: {:.3f}, 总奖励值: {:.3f}".format(
                    name_transform(room.room_type.name), 
                    event.key,
                    reward[self.selected_room_idx],
                    sum(reward)
                )
            )
            self.parse_info(info)
            self.render()
        elif event.key == 't':
            self.mode = "Increase" if self.mode == "Decrease" else "Decrease"
            print(f"当前模式: {self.mode}")
            self.render()
        elif event.key == 'a':
            self.show_room_names = not self.show_room_names
            print(f"显示房间名称: {self.show_room_names}")
            self.render()
        elif event.key == 'r':
            self.env.reset()
            self.selected_room_idx = None
            print("重置环境")
            self.render()
        elif event.key == 'q':
            plt.close()
            print("退出")

    def parse_info(self, info:Dict) -> None:
        """ 解析并打印信息 """
        rf = info['reward_components']
        name = ['boundary', 'area', 'aspect_ratio', 'edge', 'corner', 'prior_layout', 'overlap']
        name_cn = ["越界惩罚", "面积惩罚", "长宽比惩罚", "贴边奖励", "角落奖励", "先验位置惩罚", "重叠惩罚"]
        for i, room in enumerate(self.env.rooms):
            if i == self.selected_room_idx:
                for a, a_cn in zip(name, name_cn):
                    if a not in rf[i].keys():
                        continue
                    print("{}: {:.3f}".format(a_cn, rf[i][a]), end=", ")
    
    def render(self):
        """渲染环境，高亮选中的房间"""
        init_livingroom = False
        self.ax.clear()
        ax = self.ax
        
        # 绘制边界
        if self.env.boundary:
            xs, ys = zip(*self.env.boundary)
            ax.plot(xs + (xs[0], ), ys + (ys[0], ), 'k-', linewidth=3)
            
        if self.env.living_room:
            if init_livingroom:
                room = self.env.living_room
                # 房间矩形
                room_rect = patches.Rectangle(
                    (room.x1, room.y1), room.width, room.height,
                    linewidth=2, edgecolor='red', 
                    facecolor=RoomColor[room.room_type],
                    alpha=0.7
                )
                ax.add_patch(room_rect)
        
        # 绘制入口门
        if self.env.entrance_door:
            ed_x1, ed_y1, ed_x2, ed_y2 = self.env.entrance_door
            if self.env.language.lower() == "zh":
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3, label='入口门')
            elif self.env.language.lower() == "en":
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3, label='Entrance Door')
            else:
                ax.plot([ed_x1, ed_x2], [ed_y1, ed_y2], 
                    color=DoorColor, linewidth=3,)
        
        # 首先绘制实际客厅区域（作为背景）
        if self.env.boundary_polygon and self.env.rooms:
            # 创建所有房间的多边形集合（只包含其他房间，不包含客厅）
            all_room_polygons = []
            for room in self.env.rooms:
                all_room_polygons.append(room.get_polygon())
            
            # 计算实际客厅区域 = 边界 - 所有其他房间区域
            actual_living_area = self.env.boundary_polygon
            for room_poly in all_room_polygons:
                actual_living_area = actual_living_area.difference(room_poly)
            
            # 绘制实际客厅区域（浅灰色背景）
            if hasattr(actual_living_area, 'geoms'):  # MultiPolygon
                for polygon in actual_living_area.geoms:
                    if polygon.geom_type == 'Polygon':
                        x, y = polygon.exterior.xy
                        ax.fill(x, y, alpha=0.3, color='lightgray', 
                            linewidth=1, edgecolor='gray')
            elif actual_living_area.geom_type == 'Polygon':
                x, y = actual_living_area.exterior.xy
                ax.fill(x, y, alpha=0.3, color='lightgray', 
                    linewidth=1, edgecolor='gray')
        
        # 绘制其他房间（在客厅背景之上）
        for room in self.env.rooms:
            # 房间矩形
            room_rect = patches.Rectangle(
                (room.x1, room.y1), room.width, room.height,
                linewidth=2, edgecolor='black', 
                facecolor=RoomColor[room.room_type],
                alpha=0.8
            )
            ax.add_patch(room_rect)
            
            # 房间标签
            if self.show_room_names:
                if self.env.language.lower() == "zh":
                    ax.text(room.x1 + room.width/2, room.y1 + room.height/2,
                        name_transform(room.room_type.name), 
                        ha='center', va='center', fontweight='bold', fontsize=10)
                elif self.env.language.lower() == "en":
                    ax.text(room.x1 + room.width/2, room.y1 + room.height/2,
                        room.room_type.name, 
                        ha='center', va='center', fontweight='bold', fontsize=10)
            
            # 门
            if room.door_position:
                dx1, dy1, dx2, dy2 = room.door_position
                ax.plot([dx1, dx2], [dy1, dy2], 
                    color=DoorColor, linewidth=3)
        
        # 绘制先验位置（如果存在）
        if self.env.prior_positions and self.env.show_prior:
            for i, (cx, cy) in enumerate(self.env.prior_positions):
                if self.env.language.lower() == "zh":
                    ax.scatter(cx, cy, color='red', s=20, marker='x', linewidth=2, 
                        label=f'先验位置{i+1}')
                elif self.env.language.lower() == "en":
                    ax.scatter(cx, cy, color='red', s=20, marker='x', linewidth=2, 
                        label=f'Prior Position {i+1}')
        
        # 设置图形属性
        # Determine plot limits
        max_coord = self.env.grid_size
        if self.env.boundary:
            xs, ys = zip(*self.env.boundary)
            max_x = max(xs)
            max_y = max(ys)
            max_coord = max(max_coord, max_x + 2, max_y + 2) # Add some margin
            
        ax.set_xlim(0, max_coord)
        ax.set_ylim(0, max_coord)
        ax.set_aspect('equal')

        if self.env.language.lower() == "zh":
            ax.set_title('房屋布局可视化', fontsize=14, fontweight='bold')
            ax.set_xlabel('宽度 (m)')
            ax.set_ylabel('高度 (m)')
        elif self.env.language.lower() == "en":
            ax.set_title('House Layout Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel('Width (m)')
            ax.set_ylabel('Height (m)')
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加说明文本
        if self.selected_room_idx is not None:
            room = self.env.rooms[self.selected_room_idx]
            info_text = ""
            if self.env.language.lower() == "zh":
                info_text += "控制: 方向键移动, \nY/H调整高度, \nG/J调整宽度\n"
                info_text += "T切换, A开关名称, R重置, Q退出"
                
                if self.mode == "Increase":
                    self.ax.set_title('房屋布局可视化 - 使用方向键和YHGJ控制选中房间, 当前模式`增加`')
                else:
                    self.ax.set_title('房屋布局可视化 - 使用方向键和YHGJ控制选中房间, 当前模式`减少`')
            elif self.env.language.lower() == "en":
                info_text += "Controls: Use arrow keys to move, \nY/H to adjust height, \nG/J to adjust width\n"
                info_text += "T to toggle, A to toggle names, R to reset, Q to quit"

                if self.mode == "Increase":
                    self.ax.set_title('House Layout Visualization - Use arrow keys and YHGJ to control selected room, `Increase`')
                else:
                    self.ax.set_title('House Layout Visualization - Use arrow keys and YHGJ to control selected room, `Decrease`')
            
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            if self.env.language.lower() == "zh":
                self.ax.set_title('房屋布局可视化')
                # 显示各个房间的面积和宽高比 - 移动到右下角
                info_text = "房间信息:\n"
                # 各个房间的面积和长宽比
                for room in self.env.rooms:
                    info_text += f"{name_transform(room.room_type.name)}: {room.area:.2f}㎡, {room.aspect_ratio:.2f}\n"
                # info_text放到右下角
                ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                        horizontalalignment="right",
                        verticalalignment='bottom', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            elif self.env.language.lower() == "en":
                self.ax.set_title('House Layout Visualization')
                # 显示各个房间的面积和宽高比 - 移动到右下角
                info_text = "Room Info:\n"
                # 各个房间的面积和长宽比
                for room in self.env.rooms:
                    info_text += f"{room.room_type.name}: {room.area:.2f}m^2, {room.aspect_ratio:.2f}\n"
                # info_text放到右下角
                ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                        horizontalalignment="right",
                        verticalalignment='bottom', fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self.fig.canvas.draw()
        

if __name__ == "__main__":
    env, conf = make_env(CONFIG_PATH)
    env.reset()
    
    # 创建交互式环境
    interactive_env = InteractiveHouseEnv(env)
    interactive_env.render()
    
    print("交互式房屋布局环境已启动!")
    print("使用说明:")
    print("1. 点击选择房间")
    print("2. 使用方向键移动选中的房间")
    print("3. 使用Y/H键调整高度")
    print("4. 使用G/J键调整宽度")
    print("5. 使用T切换增加/减少模式")
    print("6. 按A键开关房间名称显示")
    print("7. 按R键重置环境")
    print("8. 按Q键退出")
    
    plt.show()
