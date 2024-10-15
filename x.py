<<<<<<< HEAD
# 修正可能範囲　195-214行目に記載

=======
# 修正可能範囲　119-135行目に記載
>>>>>>> 76f6b566ccc2edf954a2c7f54c9076a3794c25c1
from copy import copy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18
import os
from datetime import datetime

from modules import system as s
#from modules import plot as p

from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
<<<<<<< HEAD
        経験再生バッファの初期化
        Args:
        - buffer_size (int): バッファに保存できるデータの最大数
        - batch_size (int): バッチごとに取得するデータの数
        """
        self.buffer = deque(maxlen=buffer_size)  # 経験データを格納するデータ構造
=======
        リプレイバッファの初期化。バッファサイズとバッチサイズを指定します。

        Args:
        - buffer_size (int): リプレイバッファの最大サイズ。
        - batch_size (int): 1度に取得するサンプルの数。
        """
        self.buffer = deque(maxlen=buffer_size)  # 経験を格納するための循環バッファ
>>>>>>> 76f6b566ccc2edf954a2c7f54c9076a3794c25c1
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
<<<<<<< HEAD
        バッファに新しい経験データを追加
        Args:
        - state (np.ndarray): 現在の状態
        - action (int): 実行した行動
        - reward (float): 行動による報酬
        - next_state (np.ndarray): 次の状態
        - done (bool): エピソード終了フラグ
=======
        新しい経験をリプレイバッファに追加します。

        Args:
        - state (numpy.ndarray): 現在の環境の状態。
        - action (int): 現在の状態で取った行動。
        - reward (float): 行動を取った後に得た報酬。
        - next_state (numpy.ndarray): 行動後に観測された次の状態。
        - done (int): エピソードが終了したかどうかを示すフラグ（1: 終了、0: 継続）。
>>>>>>> 76f6b566ccc2edf954a2c7f54c9076a3794c25c1
        """
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)  # 新しい経験をバッファに追加

    def __len__(self):
        """
<<<<<<< HEAD
        バッファに格納されている経験の数を返す
        """
        return len(self.buffer)
    
    def get_batch(self):
        """
        バッファからランダムにバッチサイズ分の経験を取得
        Returns:
        - state (torch.Tensor): バッチの状態
        - action (torch.Tensor): バッチの行動
        - reward (torch.Tensor): バッチの報酬
        - next_state (torch.Tensor): バッチの次の状態
        - done (torch.Tensor): バッチのエピソード終了フラグ
        """
        data = random.sample(self.buffer, self.batch_size)
        state = torch.tensor(np.stack([x[0] for x in data])).cuda()
        action = torch.tensor(np.array([x[1] for x in data]).astype(int)).cuda()
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).cuda()
        next_state = torch.tensor(np.stack([x[3] for x in data])).cuda()
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).cuda()
=======
        リプレイバッファの現在のサイズを返します。
        """
        return len(self.buffer)  # バッファ内のデータ数を返す
    
    def get_batch(self):
        """
        リプレイバッファから経験をバッチ単位で取得します。

        Returns:
        - state (torch.Tensor): 現在の状態のバッチ。
        - action (torch.Tensor): 行動のバッチ。
        - reward (torch.Tensor): 報酬のバッチ。
        - next_state (torch.Tensor): 次の状態のバッチ。
        - done (torch.Tensor): エピソードが終了したかを示すフラグのバッチ。
        """
        data = random.sample(self.buffer, self.batch_size)  # バッファからランダムにサンプルを取得
        state = torch.tensor(np.stack([x[0] for x in data])).cuda()  # 状態をテンソルに変換
        action = torch.tensor(np.array([x[1] for x in data]).astype(int)).cuda()  # 行動をテンソルに変換
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32)).cuda()  # 報酬をテンソルに変換
        next_state = torch.tensor(np.stack([x[3] for x in data])).cuda()  # 次の状態をテンソルに変換
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32)).cuda()  # 終了フラグをテンソルに変換
>>>>>>> 76f6b566ccc2edf954a2c7f54c9076a3794c25c1
        return state, action, reward, next_state, done


#　ニューラルネットワーク
class QNet(nn.Module):
    def __init__(self, action_size):
        """
        Qネットワークの初期化
        4層の全結合層を持つネットワーク
        Args:
        - action_size (int): 行動のサイズ（行動の選択肢数）
        """
        super().__init__()
        self.l1 = nn.Linear(6, 128)  # 入力は６次元（状態数）
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, action_size)  # 出力は行動数

    def forward(self, x):
        """
        ネットワークのフォワードパス
        Args:
        - x (torch.Tensor): 入力テンソル（状態）
        Returns:
        - torch.Tensor: 行動価値の予測結果
        """
        x = x.float()  # データ型の変換
        x = F.relu(self.l1(x))  # 隠れ層１
        x = F.relu(self.l2(x))  # 隠れ層２
        x = F.relu(self.l3(x))  # 隠れ層３
        x = self.l4(x)  # 出力層（行動価値）
        return x

#　エージェント
class DQNAgent:
    def __init__(self):
        """
        DQNエージェントの初期化
        強化学習に必要なパラメータを設定し、Qネットワークを初期化
        """
        self.gamma = 0.98  # 割引率
        self.lr = 0.001  # 学習率
        self.lr_decay = 0.9999  # 学習率の減衰率
        self.buffer_size = 10000  # 経験再生のバッファサイズ
        self.batch_size = 32  # ミニバッチのサイズ
        self.action_size = 3  # 行動の選択肢

        # リプレイバッファ、Qネットワーク、ターゲットネットワーク、オプティマイザ、学習率スケジューラを初期化
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).cuda()  # Qネットワーク
        self.qnet_target = QNet(self.action_size).cuda()  # ターゲットネットワーク
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)

    def sync_qnet(self):
        """
        Qネットワークのパラメータをターゲットネットワークにコピー
        """
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state, epsilon):
        """
        行動選択
        ε-greedy法に基づき、ランダムまたはQネットワークの予測に基づいて行動を選択
        Args:
        - state (np.ndarray): 現在の状態
        - epsilon (float): 探索率（ランダム行動を選ぶ確率）
        Returns:
        - int: 選択された行動
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)  # ランダムな行動選択
        else:
            state = torch.tensor(state[np.newaxis, :]).cuda()  # 状態をテンソルに変換
            qs = self.qnet(state)  # Qネットワークで行動価値を予測
            return qs.argmax().item()  # 最大の行動価値を持つ行動を選択

    def update(self, state, action, reward, next_state, done):
        """
        経験再生バッファにデータを追加し、Qネットワークの重みを更新
        Args:
        - state (np.ndarray): 現在の状態
        - action (int): 実行した行動
        - reward (float): 得た報酬
        - next_state (np.ndarray): 次の状態
        - done (bool): エピソード終了フラグ
        Returns:
        - loss.data (torch.Tensor): 損失関数
        """
        # バッファに経験を追加
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None  # バッファが十分にたまっていない場合は更新しない

        # バッファから経験を取得
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        state = state.cuda()
        action = action.cuda()
        reward = reward.cuda()
        next_state = next_state.cuda()
        done = done.cuda()

        # Qネットワークとターゲットネットワークの予測値を計算
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]
        next_q.detach()

        # TDターゲットを計算
        target = reward + (1-done) * self.gamma * next_q
#        target = reward + self.gamma * next_q

        # 損失を計算し，バックプロパゲーションでパラメータを更新
        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data


def main():
<<<<<<< HEAD
    start_time = datetime.now()  # 処理開始時間

    # 以下修正可能 ------------------------------------------------------------------------
=======
    start_time = datetime.now()  # 処理開始時刻
    # 以下修正可能--------------------------------------------------------------------------
    episodes = 2000              # エピソード数
    record = episodes/50         # 記録間隔
    sync_interval = episodes/10  #　同期間隔
    directory = "H=x_dH=10_da=0.01_ani=(0,0,100)"   # ファイル名
    os.mkdir(directory)

    t_limit = 2e-9 # [s]         # 終了時間
    dt = t_limit / 1e3 # [s]
    limit = int(t_limit / dt)
    alphaG = 0.01                # ギルバート減衰定数
    anisotropy = np.array([0e0, 0e0, 100e0]) # [Oe]    # 異方性
    H_shape = np.array([0.012*10800*0, 0.98*10800*0, 0.008*10800*0])  # [Oe]   # 反磁場
    dh = 10 # [Oe]   # 行動間隔ごとの磁場変化
    da = 1e-11 # [s]   # 行動間隔  da<=1e-10
    m0 = np.array([0e0, 0e0, 1e0])  #  初期磁化
    # 以上修正可能--------------------------------------------------------------------------
    b = 0
>>>>>>> 76f6b566ccc2edf954a2c7f54c9076a3794c25c1

    episodes = 2000  # エピソード数
    record = episodes/50  # 結果の記録間隔
    sync_interval = episodes/10  #　ターゲットネットワークを同期する間隔
    directory = "H=x_dH=10_da=0.01_ani=(0,0,100)"   # 結果を保存するディレクトリ名
    os.mkdir(directory)  # ディレクトリを作成

    # シミュレーション設定
    t_limit = 2e-9  # シミュレーションの終了時間 [秒]
    dt = t_limit / 1e3  # タイムステップ [秒]
    limit = int(t_limit / dt)  # ステップ数
    alphaG = 0.01  # ギルバート減衰定数
    anisotropy = np.array([0e0, 0e0, 100e0])  # 異方性定数 [Oe]
    H_shape = np.array([0.012*10800*0, 0.98*10800*0, 0.008*10800*0])  # 反磁場 [Oe]
    dh = 10  # １回の行動での磁場変化 [Oe]
    da = 1e-11 # [s]  # 行動間隔 [秒]  da<=1e-10
    m0 = np.array([0e0, 0e0, 1e0])  # 初期磁化

    # 以上修正可能 ------------------------------------------------------------------------

    b = 0  # 磁化の接線の切片

    agent = DQNAgent()  # DQNエージェントの初期化
    reward_history = []  # 報酬の履歴を保存するリスト
    best_reward = -500  # 最良報酬の初期値
#    loss_history = []

    # 各エピソードごとにシミュレーション実行
    for episode in range(episodes):
        print("episode:{:>4}".format(episode+1), end=":")  # エピソード番号を表示
        dynamics = s.Dynamics(dt, alphaG, anisotropy, H_shape, m0)  # シミュレーションの初期化

        # 各種リストの初期化（時間，磁化，外部磁場，異方性磁場，反磁場）
        t = []
        m = []
        h = []
        Hani = []
        Hshape = []

        #ε-greedy方策のεをエピソードに応じて変化させる
#        epsilon = 0.1
        if episode < episodes/2:
            epsilon = (0.1-1)/(episodes/2-0)*(episode) + 1
        else:
            epsilon = 0.1

        old_m = np.array([0e0, 0e0, 1e0])  # 前の磁化ベクトル
        old_mz = m0[2]  # 磁化のz成分の前の値
        max_slope = -0.01  # 磁化の変化率の最大値
        reward = 0  # 報酬
        total_reward = 0  # エピソード内の総報酬
        total_loss = 0  # 損失の合計
        cnt = 0  # 損失を計算するステップ数
        done = 0  # エピソードが終了したかどうか
        field = np.array([0e0, 0e0, 0e0])  # 外部磁場
        t.append(0)  # 時刻リストに初期値を追加
        m.append(old_m)  # 磁化リストに初期値を追加
        h.append(copy(field))  # 外部磁場リストに初期値を追加
        Hani.append(anisotropy*old_m)  # 異方性磁場リストに初期値を追加
        Hshape.append(H_shape*old_m)  # 反磁場リストに初期値を追加
        old_state = np.concatenate([old_m, field])  # 状態を連結して作成

        # シミュレーションのメインループ
        for i in range(1, limit+1):
            if i == 1:
                action = agent.get_action(old_state, epsilon)  # ε-greedy方策に基づいて行動を選択
                h0 = action - 1
                old_action = action                    

            field += np.array([dh*h0*dt/da, 0e0, 0e0])  # 外部磁場を更新

            time = i*dt  # 時刻を更新

            dynamics.RungeKutta(field)  # 磁化の時間発展を計算

#            if i % 10 == 0:
            t.append(time)  # 時刻を保存
            m.append(dynamics.m)  # 磁化を保存
            h.append(copy(field))  # 外部磁場を保存
            Hani.append(anisotropy*dynamics.m)  # 異方性磁場を保存
            Hshape.append(H_shape*dynamics.m)  # 反磁場を保存

            # 磁化の接線の式を計算
            slope = (dynamics.m[2]-old_mz) / dt
            old_mz = dynamics.m[2]
            if slope < max_slope:
                max_slope = slope
                b = dynamics.m[2] - slope*time 

            if i % (da/dt) == 0:  # 行動間隔に達したら
                state = np.concatenate([dynamics.m, field/1e4])  # 新しい状態を作成
                action = agent.get_action(state, epsilon)  # 次の行動を選択
                h0 = action - 1

                reward = - dynamics.m[2]**3  # 報酬を計算
                if field[0] == 0:
                    reward *= 1.08  # Hx=0のとき報酬を増加
                total_reward += reward  # 合計報酬を計算

                if i == limit:
                    done = 1  # エピソードが終了したかを判定
                             
                # DQNエージェントを更新
                loss = agent.update(old_state, old_action, reward, state, done)

                if episode > sync_interval:
                    total_loss += loss  # 損失を集計
                    cnt += 1

                old_state = state  # 状態を更新
                old_action = action  # 行動を更新
                reward = 0  # 報酬をリセット

        if episode % sync_interval == 0:
            agent.sync_qnet()  # ターゲットネットワークを同期

        agent.scheduler.step()  # 学習率を調整

        if episode % record == record-1:
            s.save_episode(episode+1, t, m, h, directory)  # 結果を保存

        # 最良のエピソードを保存
        if total_reward > best_reward:
            best_episode = episode + 1
            best_reward = total_reward
            best_m = np.array(m)
            best_h = np.array(h)
            best_Hani = np.array(Hani)
            best_Hshape = np.array(Hshape)
            best_slope = max_slope
            best_b = b
            reversal_time = (-1-best_b)/best_slope  # 磁化反転時間を計算

#        if episode == 100-1:
#            one_m = np.array(m)
#            one_h = np.array(h)

#        if episode == 200-1:
#            two_m = np.array(m)
#            two_h = np.array(h)

        print("reward = {:.9f}".format(total_reward))  # 合計報酬を表示

        reward_history.append(total_reward)  # 報酬履歴を保存

#        if episode > sync_interval:
#            ave_loss = total_loss / cnt
#            loss_history.append(ave_loss)
#            s.save_loss_history(loss_history, directory)

        s.save_reward_history(reward_history, directory)  # 報酬履歴をグラフに

    s.save_episode(0, t, best_m, best_h, directory)  # 最良のエピソードを保存

    end_time = datetime.now()  # 処理終了時刻
    duration = end_time - start_time  # 処理時間を計算

    # 最良のエピソードの磁化をプロット
    x = np.linspace(0, t_limit, 1000)
    y = best_slope*x + best_b
    plt.figure(figsize=(6,6))
    plt.ylim(-1, 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Magnetization')
    plt.plot(np.array(t), best_m[:,0], label='$m_z$')
    plt.plot(np.array(t), best_m[:,1], label='$m_z$')
    plt.plot(np.array(t), best_m[:,2], label='$m_z$')
    plt.legend(fontsize=16)
    plt.plot(x, y, color='darkgreen', linestyle='dashed')
    plt.tight_layout()
    plt.savefig(directory+"/reversal_time.png", dpi=200)
    plt.close()


    # 最良のエピソードの磁場をプロット
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    y_max = best_h.max()
    if best_h.max() <= best_Hani.max():
        y_max = best_Hani.max()
    if best_Hani.max() <= best_Hshape.max():
        y_max = best_Hshape.max()
    y_min = best_h.min()
    if best_h.min() >= best_Hani.min():
        y_min = best_Hani.min()
    if best_Hani.min() >= best_Hshape.min():
        y_min = best_Hshape.min()

    axes[0].set_ylim([y_min, y_max])
    axes[0].plot(t, best_h[:,0], label='$h_x$')
    axes[0].plot(t, best_h[:,1], label='$h_y$')
    axes[0].plot(t, best_h[:,2], label='$h_z$')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('$H_{\mathrm{ext}}$ [Oe]')
    axes[0].legend()

    axes[1].set_ylim([y_min, y_max])
    axes[1].plot(t, best_Hani[:,0], label='$h_x$')
    axes[1].plot(t, best_Hani[:,1], label='$h_y$')
    axes[1].plot(t, best_Hani[:,2], label='$h_z$')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('$H_{\mathrm{ani}}$ [Oe]')
    axes[1].legend()

    axes[2].set_ylim([y_min, y_max])
    axes[2].plot(t, best_Hshape[:,0], label='$h_x$')
    axes[2].plot(t, best_Hshape[:,1], label='$h_y$')
    axes[2].plot(t, best_Hshape[:,2], label='$h_z$')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('$H_{\mathrm{shape}}$ [Oe]')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(directory+"/field.png", dpi=200)
    plt.close()


#    p.plot_energy(m_max, dynamics)
#    p.plot_3d(best_m)

    # 結果をtxtファイルに保存（磁化，磁場，タイムステップ，報酬）
    np.savetxt(directory+"/m.txt", best_m)
    np.savetxt(directory+"/h.txt", best_h)
    np.savetxt(directory+"/t.txt", t)
#    np.savetxt(directory+"/100th_m.txt", one_m)
#    np.savetxt(directory+"/100th_h.txt", one_h)
#    np.savetxt(directory+"/200th_m.txt", two_m)
#    np.savetxt(directory+"/200th_h.txt", two_h)
    np.savetxt(directory+"/reward history.txt", reward_history)

    # 設定と結果をオプションファイルに保存
    with open(directory + "/options.txt", mode='w') as f:
        f.write(f"""
    alphaG = {alphaG}
    anisotropy = {anisotropy} [Oe]
    H_shape = {H_shape} [Oe]
    dH = {dh} [Oe]
    da = {da} [s]
    m0 = {m0}
    time limit = {t_limit} [s]

    best episode = {best_episode}
    reversal time = {reversal_time} [s]
    processing time = {duration}
    
    slope = {best_slope}
    segment = {best_b}
    average reward = {best_reward / (t_limit / da)}
    """)



if __name__ == '__main__':
    main()
