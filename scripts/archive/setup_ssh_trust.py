"""在第二台机器上执行：配置两台机器之间的 SSH 免密互信（通用模板）。

拓扑：
  本机 = node2（可通过 SSH 访问 node1）
  远端 = node1

步骤：
  1. 本机生成 ed25519 密钥（若无）
  2. 通过内网在 node1 上也生成 ed25519 密钥（若无）
  3. 把 node2 的公钥追加到 node1 的 authorized_keys
  4. 把 node1 的公钥追加到 node2 的 authorized_keys
  5. 双向免密验证

用法（密码不要写死在脚本里，从环境变量读取）：
  SSH_PASS='<node1 密码>' NODE1_IP='<node1 内网 IP>' \\
  NODE2_IP='<本机内网 IP>' python scripts/setup_ssh_trust.py
"""

import os
import sys
import subprocess

import paramiko

NODE1_IP = os.environ.get("NODE1_IP", "")
NODE2_IP = os.environ.get("NODE2_IP", "")


def get_password() -> str:
    pwd = os.environ.get("SSH_PASS", "")
    if not pwd:
        sys.exit("缺少 SSH_PASS 环境变量（node1 的 SSH 密码），不要硬编码凭据")
    return pwd


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def ensure_key(ssh_cmd_prefix: str, remote=False) -> str:
    """确保 SSH 密钥存在，返回公钥内容。remote=False 在本机执行。"""
    if remote:
        # 远端通过 paramiko 执行
        raise NotImplementedError
    if not os.path.isfile("/root/.ssh/id_ed25519.pub"):
        os.makedirs("/root/.ssh", exist_ok=True)
        sh("ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519")
    return open("/root/.ssh/id_ed25519.pub").read().strip()


def remote_exec(ip: str, cmd: str) -> tuple[str, str]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username=os.environ.get("SSH_USER", "root"),
              password=get_password(), timeout=20)
    _, out, err = c.exec_command(cmd, timeout=60)
    o = out.read().decode()
    e = err.read().decode()
    c.close()
    return o, e


def main():
    print(f"== 本机是 node2（{NODE2_IP}） ==")
    # 1. 本机生成密钥
    pub2 = ensure_key("")
    print("node2 公钥:", pub2)

    # 2. node1 生成密钥（通过 paramiko 内网连接）
    o, e = remote_exec(NODE1_IP, "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
                                 "test -f /root/.ssh/id_ed25519.pub || "
                                 "ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519; "
                                 "cat /root/.ssh/id_ed25519.pub")
    pub1 = o.strip().splitlines()[-1]
    print("node1 公钥:", pub1)

    # 3. node2 公钥 -> node1 authorized_keys
    o, e = remote_exec(NODE1_IP, f"mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
                                  f"grep -qF '{pub2}' /root/.ssh/authorized_keys 2>/dev/null || "
                                  f"echo '{pub2}' >> /root/.ssh/authorized_keys; "
                                  f"chmod 600 /root/.ssh/authorized_keys; "
                                  f"echo done")
    print("推送 node2 公钥到 node1:", o.strip(), e.strip())

    # 4. node1 公钥 -> node2 authorized_keys
    if not os.path.isfile("/root/.ssh/authorized_keys"):
        sh("mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys")
    with open("/root/.ssh/authorized_keys") as f:
        content = f.read()
    if pub1 not in content:
        with open("/root/.ssh/authorized_keys", "a") as f:
            f.write(pub1 + "\n")
    sh("chmod 600 /root/.ssh/authorized_keys")
    print("node1 公钥已加入 node2 authorized_keys")

    # 5. 双向验证
    print("\n== 验证 node2 -> node1 免密 ==")
    r = sh("ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
           f"root@{NODE1_IP} 'hostname'")
    print("结果:", r or "失败")

    print("\n== 验证 node1 -> node2 免密 ==")
    o, e = remote_exec(NODE1_IP, "ssh -o BatchMode=yes -o StrictHostKeyChecking=no "
                                 f"root@{NODE2_IP} 'hostname'")
    print("结果:", o.strip() or e.strip() or "失败")


if __name__ == "__main__":
    main()
