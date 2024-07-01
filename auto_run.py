import os
import re
import sys
import time

CMD = 'python -u'

PWD = os.getcwd()
SCRIPT = 'train_vanilla.py'
# WORKSPACE = os.path.join(PWD, WORKSPACE)
WORKSPACE = PWD

PARAMETERS = f'--owner DaiJian --cfg ./cfgs/training/2hao_Dphm.toml'

THRESHOLD = 6000
THRESHOLD = 20000

CHECK_REG = r'^.*--owner DaiJian.*$'

SCREEN_SESSION = 'djrunner'
NUM_GPUS = 1


def prepare_screen_session():
    """Check if the assigned screen session is exist."""
    all_sessions = os.popen('screen -ls').readlines()
    is_exist = False
    for each in all_sessions:
        each = each.strip()
        match = re.match(fr'^[\d\\.]+{SCREEN_SESSION}\s+.*$', each)
        if match:
            is_exist = True
            break
    if not is_exist:
        # Create new session
        os.system(f'screen -dmS {SCREEN_SESSION}')
    os.system(f"screen -S {SCREEN_SESSION} -p 0 -X stuff 'conda activate py11\n'")
    os.system(f"screen -S {SCREEN_SESSION} -p 0 -X stuff 'cd {WORKSPACE}\n'")
    pass


if __name__ == '__main__':
    start_time = time.time()
    hours = 0
    while True:
        if (time.time() - start_time) // 3600 >= hours + 1:
            hours += 1
            print(f'Already wait {hours} h.')
        gpu_info = os.popen('nvidia-smi | grep %').readlines()
        for i in range(len(gpu_info)):
            time.sleep(1)
            gpu_status = gpu_info[i].split('|')
            # 第三个为显存信息
            memory = gpu_status[2].strip()
            match = re.match(r'^(\d+)\D+(\d+)\D+$', memory)
            used = int(match.group(1))
            total = int(match.group(2))
            if total - used > THRESHOLD:
                print(f'Free GPU {i}')
                prepare_screen_session()
                if NUM_GPUS >= 2:
                    cuda = f'CUDA_VISIBLE_DEVICES=0,1'
                else:
                    cuda = f'CUDA_VISIBLE_DEVICES={i}'
                # screen -S dj -p 0 -X stuff "cmd\n"
                # Send a ENTER to screen to execute the command
                params = f'{PARAMETERS} -g {i}'
                # params = PARAMETERS
                cmd = f'screen -S {SCREEN_SESSION} -p 0 -X stuff \'' + ' '.join(
                    [cuda, CMD, SCRIPT, params]) + '\n' + '\''
                print(f'Send command: {cmd}')
                os.system(cmd)
                # 60s后检查进程是否成功启动
                time.sleep(30)
                process = os.popen(r'ps -ef | grep python').readlines()
                for each_process in process:
                    process_search = re.search(CHECK_REG, each_process)
                    if process_search:
                        print('Get progress')
                        sys.exit(0)
                print('Progress failed')
