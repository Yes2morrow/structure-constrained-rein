
# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import Trainer
from core import make_env
from common.project_paths import CONFIG_PATH, RESULTS_DIR

if __name__ == "__main__":
    # 创建环境
    # 需要将config.yaml替换成预训练模型中的配置文件路径，并且修改Training -> ckpt_path；添加预训练模型
    env, conf = make_env(CONFIG_PATH)
    
    # 创建训练器
    runner = Trainer(
        agent_name=conf['Training']['agent_name'],
        muliple_agents=True,
        save_dir=RESULTS_DIR)
    
    runner.evaluate(env, max_iters=conf['Training']['max_steps'],
                        pretrain_ckpt_path=conf['Training']['ckpt_path'],
                        render=True)
