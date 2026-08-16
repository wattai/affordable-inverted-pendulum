# 2026-08-15 時点のベスト設計：モデルと制御則

このディレクトリは、実機が最も安定して立った時点の構成を保存したものです。

| ファイル | 内容 |
| --- | --- |
| [design_best_R14_controller.py](design_best_R14_controller.py) | 物理パラメータから 10 状態モデルを組み、離散化して LQR ゲインを求める設計スクリプト |
| [arduino_uno_sketch.ino](arduino_uno_sketch.ino) | 実機（Arduino Uno）に書き込んだスケッチ。上記モデルの $A_d, B_d$、オブザーバゲイン $L$、LQR ゲイン $K$ が直値で埋め込まれている |

以下、記号・モデル・制御則を数式で示し、最後にコード上の実測値との対応と注意点をまとめます。

---

## 1. 記号と状態変数

| 記号 | 意味 | 単位 |
| --- | --- | --- |
| $\theta$ | 機体の傾き角（直立が 0） | rad |
| $\dot\theta$ | 機体の角速度 | rad/s |
| $\varphi$ | 左右車輪角の平均 | rad |
| $\omega$ | 車輪角速度（モータ 1 次遅れ後の値） | rad/s |
| $z_1,\dots,z_6$ | 入力むだ時間を近似する 1 次遅れ 6 段の内部状態 | rad/s |
| $u$ | 車輪速度指令（下位ループへの入力） | rad/s |

状態ベクトルは

$$
x = \begin{bmatrix}\theta & \dot\theta & \varphi & \omega & z_1 & z_2 & z_3 & z_4 & z_5 & z_6\end{bmatrix}^{\mathsf T}\in\mathbb R^{10}
$$

観測量は上位 4 成分（IMU の角度・角速度、エンコーダの角度・角速度）です。

$$
y = Cx,\qquad C = \begin{bmatrix} I_4 & 0_{4\times 6}\end{bmatrix}
$$

## 2. 物理パラメータ

| パラメータ | 記号 | 値 |
| --- | --- | --- |
| 制御周期 | $T_s$ | 0.010 s |
| 車輪半径 | $r$ | 0.027 m |
| 重力加速度 | $g$ | 9.8 m/s² |
| 機体質量 | $m$ | 0.250–0.350 kg（下記注意） |
| 重心高さ | $h$ | 0.080 m（同上） |
| 振り子周期（実測） | $T_p$ | 0.75 s |
| モータ 1 次遅れ時定数 | $\tau_m$ | 0.070 s |
| むだ時間 | $L_d$ | 0.100 s |
| むだ時間の分割段数 | $N$ | 6 |

機体の有効慣性モーメントは、実測した振り子周期 $T_p$ から逆算します。

$$
I_{\mathrm{eff}} = \frac{T_p^{2}\, m g h}{4\pi^{2}}
$$

むだ時間 1 段あたりの時定数と遅れレートは

$$
\theta_d = \frac{L_d}{N} = \frac{1}{60}\ \mathrm{s},\qquad
\lambda = \frac{1}{\theta_d} = \frac{N}{L_d} = 60\ \mathrm{s^{-1}}
$$

### 注意：$m$ と $h$ はモデルに影響しない

$I_{\mathrm{eff}}$ を上式で定義しているため、モデル行列に現れる 2 つの係数から $m,h$ が約分で消えます。

$$
a_\theta = \frac{m g h}{I_{\mathrm{eff}}} = \frac{4\pi^{2}}{T_p^{2}} = 70.183854\ \mathrm{s^{-2}}
$$

$$
a_{bw} = \frac{m h r}{I_{\mathrm{eff}}\,\tau_m} = \frac{4\pi^{2} r}{T_p^{2} g\, \tau_m} = 2.762338\ \mathrm{s^{-2}}
$$

つまりモデルは $T_p,\ r,\ g,\ \tau_m$ だけで決まります。設計スクリプトの docstring（$m=0.250$, $h=0.105$）・コード本体（$m=0.350$, $h=0.080$）・Arduino（$m=0.250$, $h=0.080$）で値が食い違っていますが、**得られる $A_c$ は完全に同一**です。

## 3. 連続時間モデル

$$
\dot x = A_c x + B_c u
$$

各行の意味は次のとおりです。

$$
\begin{aligned}
\dot\theta &= \dot\theta \\
\ddot\theta &= a_\theta\,\theta + a_{bw}\,\omega - a_{bw}\,z_1 \\
\dot\varphi &= \omega \\
\dot\omega &= -\frac{1}{\tau_m}\omega + \frac{1}{\tau_m} z_1 \\
\dot z_i &= \lambda\,(-z_i + z_{i+1}) \quad (i=1,\dots,5) \\
\dot z_6 &= \lambda\,(-z_6 + u)
\end{aligned}
$$

- 第 2 式：重力による倒立不安定項 $a_\theta\theta$ と、車輪の駆動反力による機体への逆トルク。実際に車輪を押す量は遅延後の指令 $z_1$ なので、$\omega$ と $z_1$ の差分が機体角加速度に効きます。
- 第 4 式：モータを 1 次遅れとみなし、遅延後の速度指令 $z_1$ に時定数 $\tau_m$ で追従。
- 第 5・6 式：Padé ではなく **1 次遅れ 6 段のカスケード**で $L_d = 0.1$ s のむだ時間を近似（$u \to z_6 \to \cdots \to z_1$ の向きに伝播）。

行列で書くと

$$
A_c=\begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
a_\theta & 0 & 0 & a_{bw} & -a_{bw} & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & -\tfrac{1}{\tau_m} & \tfrac{1}{\tau_m} & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & -\lambda & \lambda & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & -\lambda & \lambda & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & -\lambda & \lambda & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -\lambda & \lambda & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\lambda & \lambda\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -\lambda
\end{bmatrix},\qquad
B_c=\begin{bmatrix}0\\0\\0\\0\\0\\0\\0\\0\\0\\ \lambda\end{bmatrix}
$$

## 4. 離散化

ゼロ次ホールドで $T_s = 10$ ms 離散化します。

$$
A_d = e^{A_c T_s},\qquad B_d = \int_0^{T_s} e^{A_c s}\,ds\ B_c
$$

$$
x[k+1] = A_d x[k] + B_d u[k]
$$

得られる $A_d$（有効数字 6 桁）：

$$
A_d=\begin{bmatrix}
1.003511 & 0.010012 & 0 & 0.000132 & -0.000109 & -0.000020 & -0.000003 & 0 & 0 & 0\\
0.702660 & 1.003511 & 0 & 0.025772 & -0.019247 & -0.005332 & -0.001024 & -0.000150 & -0.000018 & -0.000002\\
0 & 0 & 1.000000 & 0.009319 & 0.000562 & 0.000103 & 0.000015 & 0.000002 & 0 & 0\\
0 & 0 & 0 & 0.866878 & 0.099396 & 0.027555 & 0.005295 & 0.000775 & 0.000092 & 0.000009\\
0 & 0 & 0 & 0 & 0.548812 & 0.329287 & 0.098786 & 0.019757 & 0.002964 & 0.000356\\
0 & 0 & 0 & 0 & 0 & 0.548812 & 0.329287 & 0.098786 & 0.019757 & 0.002964\\
0 & 0 & 0 & 0 & 0 & 0 & 0.548812 & 0.329287 & 0.098786 & 0.019757\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0.548812 & 0.329287 & 0.098786\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0.548812 & 0.329287\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0.548812
\end{bmatrix}
$$

$$
B_d^{\mathsf T}=\begin{bmatrix}0 & 0 & 0 & 0.000001 & 0.000039 & 0.000394 & 0.003358 & 0.023115 & 0.121901 & 0.451188\end{bmatrix}
$$

対角の $0.548812 = e^{-\lambda T_s} = e^{-0.6}$、$0.866878 = e^{-T_s/\tau_m}$ です。

これらの数値は Arduino のオブザーバ予測式（[arduino_uno_sketch.ino:639-718](arduino_uno_sketch.ino#L639-L718)）にそのまま直値で埋め込まれています。

> **設計上の約束事**：LQR は $A_d, B_d$ を **小数第 6 位で丸めた行列**に対して解いています（`USE_ARDUINO_6_DECIMAL_MATRICES = True`）。Arduino に写した行列と設計用行列を一致させるためで、フル精度で解くとゲインがわずかにずれます。

## 5. オブザーバ（定常カルマンフィルタ）

観測できるのは 4 成分だけなので、$z_1..z_6$ を含む全状態を推定します。ゲイン $L$ は離散リカッチ方程式から求めた定常カルマンゲインです（$Q_{\mathrm{obs}} = I_{10}$、$R_{\mathrm{obs}} = I_4$）。

$$
P = A_d P A_d^{\mathsf T} - A_d P C^{\mathsf T}\left(C P C^{\mathsf T} + R_{\mathrm{obs}}\right)^{-1} C P A_d^{\mathsf T} + Q_{\mathrm{obs}}
$$

$$
L = P C^{\mathsf T}\left(C P C^{\mathsf T} + R_{\mathrm{obs}}\right)^{-1}
$$

実機で実装されている更新則は

$$
\hat x[k] = A_d\,\hat x[k-1] + B_d\,u[k-1] + L\left(y[k] - C\,\hat x[k-1]\right)
$$

$$
L=\begin{bmatrix}
0.604600 & 0.064392 & -0.000002 & 0.000048\\
0.064392 & 0.662611 & 0.000010 & 0.000278\\
-0.000002 & 0.000010 & 0.618044 & 0.000941\\
0.000048 & 0.000278 & 0.000941 & 0.602982\\
0.005301 & -0.020367 & 0.000800 & 0.137743\\
0.003264 & -0.012777 & 0.000452 & 0.085731\\
0.001733 & -0.006923 & 0.000230 & 0.046081\\
0.000760 & -0.003108 & 0.000096 & 0.020510\\
0.000247 & -0.001037 & 0.000030 & 0.006781\\
0.000045 & -0.000195 & 0.000005 & 0.001260
\end{bmatrix}
$$

推定誤差 $e = x - \hat x$ の同次項は

$$
e[k] = (A_d - LC)\,e[k-1] \;-\; LC(A_d - I)\,x[k-1]
$$

第 1 項の固有値は最大 $0.6536$（すべて単位円内、$T_s$ 換算で約 24 ms の収束時定数）です。第 2 項は、イノベーションの基準に予測値 $\bar x[k] = A_d\hat x[k-1] + B_d u[k-1]$ ではなく 1 サンプル前の $\hat x[k-1]$ を使っているために現れる項で、$A_d - I = O(T_s)$ なので小さいものの、厳密には教科書的な current estimator とは一致しません。実機はこの形で安定に動いています。

推定車輪速度は $\pm 20\cdot 2\pi$ rad/s に飽和させています。

## 6. LQR（バランス制御器）

評価関数

$$
J = \sum_{k=0}^{\infty}\left( x[k]^{\mathsf T} Q\, x[k] + u[k]^{\mathsf T} R\, u[k]\right)
$$

離散代数リカッチ方程式

$$
P = A_d^{\mathsf T} P A_d - A_d^{\mathsf T} P B_d\left(R + B_d^{\mathsf T} P B_d\right)^{-1} B_d^{\mathsf T} P A_d + Q
$$

最適ゲイン（離散系の正しい形）

$$
K = \left(R + B_d^{\mathsf T} P B_d\right)^{-1} B_d^{\mathsf T} P A_d,\qquad u[k] = -K\,\hat x[k]
$$

重みは

$$
Q = \mathrm{diag}(10,\ 1,\ 0,\ 1,\ 0,0,0,0,0,0),\qquad R = 1.0
$$

$Q$ の第 3 要素（車輪角 $\varphi$）を 0 にしているのがポイントで、これにより $K_\varphi = 0$ となり、位置制御を切った挙動になります。得られる全ゲインは

$$
K_{\mathrm{full}} = \begin{bmatrix}-350.441115 & -41.839831 & 0 & -5.073251 & 0.632886 & 0.557126 & 0.489127 & 0.426907 & 0.370160 & 0.318175\end{bmatrix}
$$

### 実機に載せた制御則

実験の結果、$z_1..z_6$ を直接フィードバックするより**上位 4 状態だけを使う**方がはるかに安定でした。したがって実装は

$$
u[k] = -K_\theta\hat\theta - K_{\dot\theta}\hat{\dot\theta} - K_\varphi\hat\varphi - K_\omega\hat\omega
$$

$$
K_\theta = -350.441115,\quad K_{\dot\theta} = -41.839831,\quad K_\varphi = 0,\quad K_\omega = -5.073251
$$

$$
\Rightarrow\quad u[k] \approx 350.44\,\hat\theta + 41.84\,\hat{\dot\theta} + 5.07\,\hat\omega \quad [\mathrm{rad/s}]
$$

（$\hat\theta$ には後述の位置ホールド用トリムが引かれますが、既定では無効です。）

### 閉ループ極

| 構成 | 最大 $|\lambda|$ |
| --- | --- |
| 全 10 状態 LQR（$K_{\mathrm{full}}$） | 1.000000 |
| 実機構成（$z$ ゲインを 0 に切り落とした 4 状態） | 1.080014 |
| オブザーバ誤差 $A_d - LC$ | 0.653639 |

全状態 LQR の最大極が 1.0 ちょうどなのは、$Q_\varphi = 0$ により車輪角 $\varphi$ が積分器としてコスト評価から外れているためです（$\varphi$ は可安定だが不可検出、モードは中立）。

4 状態版の $1.08$ は、**この 10 状態公称モデル単体では発散する**ことを意味します。設計スクリプト自身もこれを警告として出力します。ただし実機にはオブザーバ、下位のモータ／PWM ループ、エンコーダ特性、飽和、モデル誤差が加わっており、実験的には $z$ を直接フィードバックする構成より 4 状態構成の方が明確に良好でした。このスクリプトは「理論上の全状態 LQR」と「実際に成功した実装ゲイン」の両方を再現するためにこの形になっています。

## 7. 下位ループ（左右同期 + leaky PI）

上位が出した速度指令 $u$ を、左右 2 輪の PI ループへ渡します。$\varphi_A,\varphi_B$ を左右の車輪角、$\omega_A,\omega_B$ を左右の車輪速度として

$$
\Delta = \varphi_B - \varphi_A
$$

$$
e_A = u - \omega_A + k_{\mathrm{sync}}\Delta,\qquad
e_B = u - \omega_B - k_{\mathrm{sync}}\Delta
$$

積分項は漏れ（leaky）付きで、$\rho = 0.99$（時定数 $\approx T_s/(1-\rho) = 1$ s）：

$$
I_A[k] = e_A[k]\,T_s + \rho\, I_A[k-1],\qquad I_B[k] = e_B[k]\,T_s + \rho\, I_B[k-1]
$$

$$
u_A = \mathrm{sat}_{[-1,1]}\!\left(k_{pA} e_A + k_{iA} I_A\right),\qquad
u_B = \mathrm{sat}_{[-1,1]}\!\left(k_{pB} e_B + k_{iB} I_B\right)
$$

| パラメータ | 値 |
| --- | --- |
| $k_{\mathrm{sync}}$ | 1.20 |
| $k_{pA},\ k_{pB}$ | 0.025, 0.030 |
| $k_{iA},\ k_{iB}$ | 0.040, 0.050 |
| 積分漏れ $\rho$ | 0.99 |

左右で $k_p, k_i$ が非対称なのは、安定していたコードの実測値をそのまま残しているためです。$u_A, u_B$ は duty 比としてそのまま PWM に出力されます。

## 8. センサ処理と保護

- **機体角**：BNO055 の Euler-z からオフセット（起動時 300 サンプルの平均）を引いて rad へ変換。$|z| < 20^\circ$ の領域は Euler 表現が乱れるため、前回値を保持する処理を意図的に残しています。
- **機体角速度**：BNO055 ジャイロ $-\dot x$ をそのまま使用。
- **車輪角**：A/B 相を高速ポーリングして計数。ギア比が左右で異なります。

$$
\varphi_A = \mathrm{cnt}_A\cdot\frac{20}{14}\cdot\frac{2\pi}{48},\qquad
\varphi_B = \mathrm{cnt}_B\cdot\frac{20}{12}\cdot\frac{2\pi}{48}
$$

$$
\varphi = \frac{\varphi_A + \varphi_B}{2},\qquad \omega = \frac{\bar\omega_A + \bar\omega_B}{2}
$$

- **車輪速度の生値**：1 次 IIR で平滑（$\alpha = 0.8$）。
- **転倒保護**：$|\theta| > 35^\circ$ でモータ停止・積分器リセット。
- **タイミング**：MsTimer2 による 10 ms 周期（100 Hz）。ログは 1/5 に間引いて 20 Hz。

## 9. 実装上の注意（既知の差異）

実機の挙動を再現するうえで重要な、コードとモデルの食い違いです。**いずれも「安定していた状態を壊さない」ために意図的に、あるいは結果的に残されています。**

1. **`angleSpeedAverageA/B` は恒等的に 0**
   [arduino_uno_sketch.ino:572-578](arduino_uno_sketch.ino#L572-L578) の更新は `angleSpeedAverageA += angleSpeedAverageA / 100` という自己参照で、初期値 0・毎制御周期末に 0 リセットされるため、値は常に 0 のままです。結果として
   - オブザーバの第 4 観測 $y_4 \equiv 0$（イノベーション $e_4 = -\hat\omega$ になる）
   - 下位 PI の速度フィードバックが実質無効化され、$e_A = u + k_{\mathrm{sync}}\Delta$ となる

   コード中のコメントでも「下位ループを一から再調整しない限り直すな」と明記されています。

2. **`MODE_R14_4STATE` に入っている数値は $R = 1.4$ ではなく $R = 1.0$ の解**
   ファイル名・モード名は R14 ですが、実際に有効な定数 $(-350.441115,\ -41.839831,\ 0,\ -5.073251)$ は $Q_\varphi = 0,\ R = 1.0$ の厳密解と桁まで一致します（ソース中のコメント `// actually it assumes R=1.0` のとおり）。コメントアウトされている $(-335.154543,\ -40.013031,\ -0.765166,\ -4.857388)$ が $R = 1.4$ の解です。
   一方、`MODE_R16_4STATE` の値と参考用の `gainZ1..gainZ6` は $Q_\varphi = 1$ で設計された別系統の値です。本 README の記載は**実際に有効な定数**に合わせています。

3. **設計スクリプトの再現チェックは現状 FAIL する**
   `EXPECTED_R14_FULL` は $Q_\varphi = 1,\ R = 1.4$ 相当の値ですが、スクリプト本体は $Q_\varphi = 0$（`Q_WHEEL_ANGLE = 0.0`）で計算するため、`WARNING: gain does not match the expected R=1.4 design.` が出ます。上に書いたとおり、実機に載っているのは $Q_\varphi = 0$ 系の値なので、モデル・実装としては整合しています。

4. **$B_d$ の $z_2$ 成分の微差**
   計算値 $0.000394$ に対し Arduino は $0.000390$ を使っています（[arduino_uno_sketch.ino:714](arduino_uno_sketch.ino#L714)）。同様に $L$ の (1,3) 成分の符号が計算値 $-0.000002$ に対し Arduino は $+0.000002$ です。いずれも他成分より 3 桁以上小さく、実質的な影響はありません。

5. **位置ホールド外ループは既定で無効**
   `ENABLE_SLOW_POSITION_HOLD = false`。有効にすると 10 Hz で機体角の目標にトリムを加えます。

$$
\theta_{\mathrm{trim}}[k] = \alpha_p\,\theta_{\mathrm{trim}}[k-1] + (1-\alpha_p)\,\mathrm{sat}_{\pm 0.5^\circ}\!\left(-K_{\mathrm{pos}}(\varphi - \varphi_{\mathrm{ref}})\right)
$$

$$
\theta_{\mathrm{ctrl}} = \hat\theta - \theta_{\mathrm{trim}},\qquad K_{\mathrm{pos}} = 0.0010,\ \ \alpha_p = 0.90
$$

## 10. 再現方法

```bash
pip install numpy scipy
python scripts/bests/at-2026-08-15/design_best_R14_controller.py
```

物理パラメータ → $A_c, B_c$ → ZOH 離散化 → 6 桁丸め → 離散 LQR → Arduino 用定数ブロック、の順に出力されます。$R$ を 1.0〜1.8 で振ったスイープも最後に表示されます。

オブザーバゲイン $L$ はこのスクリプトでは計算していません。[../../system-design/calc-optimal-gain-v6.py](../../system-design/calc-optimal-gain-v6.py) の `calculate_discrete_observer_gain(A_d, C_d, I_10, I_4)` が上記 $L$ を再現します。
