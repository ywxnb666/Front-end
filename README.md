# Remote Clone EXE

一个可维护的桌面工具骨架，用于连接远程服务器，并在指定远程项目目录下 clone 固定仓库：

```text
https://gitee.com/yang-wenxiao-111/CISCN-MICAD.git
```

## 结构

```text
remote_clone_exe/
  app/
    __init__.py
    config.py
    models.py
    ui/
      __init__.py
      main_window.py
    services/
      __init__.py
      git_service.py
      ssh_client.py
  main.py
  requirements.txt
  build_exe.bat
```

## 输入规则

界面只要求输入 3 项：

- `服务器地址`
- `服务器连接命令`
- `项目地址`
- `密码`

其中 `服务器连接命令` 需要写成 ssh 连接命令，例如：

```text
ssh root@192.168.1.10
ssh ubuntu@example.com -p 22
```

## 功能

- 连接远程 Linux 服务器
- 自动创建远程项目目录
- 在远程目录下 clone 固定仓库
- 如果仓库已存在，则直接提示
- 本地保存最近一次的服务器连接命令和项目地址

## 运行

```powershell
cd D:\desktop\lord\MICAD\remote_clone_exe
pip install -r requirements.txt
python main.py
```

## 打包为 EXE

```powershell
cd D:\desktop\lord\MICAD\remote_clone_exe
build_exe.bat
```

打包完成后，生成的单文件 EXE 会位于：

```text
dist\RemoteCloneTool.exe
```

## 后续扩展建议

后续如果要继续加功能，建议沿着以下边界扩展：

- 新增远程操作逻辑：放到 `app/services/`
- 新增界面区域：放到 `app/ui/`
- 新增配置项：放到 `app/config.py`
- 新增任务状态或结果模型：放到 `app/models.py`

这样可以避免把 UI、SSH、业务逻辑揉在一个文件里。
