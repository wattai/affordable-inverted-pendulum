# 倒立振子ロボ：車輪速度フィードバックが実質的に働いていない件の整理

## 1. この文書の目的

この文書は、自走式倒立振子ロボの制御実験において、

> **「実機コードでは車輪速度フィードバックがほぼ働いていないにもかかわらず、その状態の方が姿勢追従性が良かった」**

という現象について、これまでの議論・コード解析・実機結果を1つにまとめたものです。

他のAIや制御設計者へ引き継ぐため、以下を整理します。

- 現在の実機制御構造
- `angleSpeedAverageA/B` の実装上の問題
- 実際の下位制御が何をしているか
- 理論モデルとの食い違い
- なぜ4ゲインだけの方が追従性が良い可能性があるか
- `z1..z6` 直接フィードバックや Δu-LQR が悪化した理由
- 現時点で最も良かった設計
- 今後の設計上の注意

---

# 2. 対象ロボットの概要

## ハードウェア

- Arduino Uno R3
- BNO055 IMU
- 左右2輪
- DCモータ
- エンコーダあり
- PWM駆動
- 左右独立モータ
- 車輪半径：約 `0.027 m`

## 主な物理パラメータ

これまで設計に使っていた代表値：

```text
Ts              = 0.010 s
m_body          = 0.250 kg
h_body          = 0.105 m   （後に実測では 5〜7 cm 程度と判明）
T_pendulum      = 0.75 s    （後に動画計測では約 0.45 s の可能性）
tau_motor       = 0.07 s
dead time       = 約 0.10〜0.11 s
```

現在のArduinoモデルに埋め込まれている6段遅延系の係数は、おおむね

```text
dead time ≈ 0.10 s
```

を6段一次遅れで近似したものに相当する。

---

# 3. 上位LQRの状態

設計モデルでは、以下の10状態を使用していた。

\[
x =
[\theta,\dot{\theta},\phi,\omega,z_1,z_2,z_3,z_4,z_5,z_6]^T
\]

意味：

- \(\theta\)：車体角度
- \(\dot{\theta}\)：車体角速度
- \(\phi\)：平均車輪角度
- \(\omega\)：平均車輪角速度
- \(z_1 \ldots z_6\)：入力デッドタイム近似用の内部状態

理論上の10状態LQRでは、

\[
u = -Kx
\]

なので、

\[
u =
-K_\theta \theta
-K_{\dot{\theta}}\dot{\theta}
-K_\phi \phi
-K_\omega \omega
-\sum_{i=1}^{6}K_{z_i}z_i
\]

となる。

しかし、**実機で最も追従性が良かった制御では、`z1..z6` は直接フィードバックに使っていない。**

実際に使っていたのは主に次の4状態：

\[
\boxed{
u =
-K_\theta \hat{\theta}
-K_{\dot{\theta}}\hat{\dot{\theta}}
-K_\phi \hat{\phi}
-K_\omega \hat{\omega}
}
\]

---

# 4. 現在ベストだったLQRゲイン

`R=1.4` のとき、実機で非常に良かった4ゲインは以下。

```cpp
K_theta       = -359.683463
K_theta_dot   =  -42.940531
K_wheel_angle =   -0.765166
K_wheel_speed =   -5.268757
```

この構造は、

- 姿勢追従性が良い
- 以前よりギコギコが低減
- `R=1.2` よりさらに良い
- Δu-LQR より遥かに良い
- `z1..z6` を直接制御に入れた版より遥かに良い

という実機結果だった。

---

# 5. 問題の下位制御

上位LQRの出力は、

```cpp
u_WheelAngleSpeed
```

という名前になっており、見た目上は「目標車輪角速度」に見える。

その後、左右車輪に対して次のような誤差を作っている。

```cpp
float eA =
    u_WheelAngleSpeed
    - angleSpeedAverageA
    + kWheelSync * (angleB - angleA);

float eB =
    u_WheelAngleSpeed
    - angleSpeedAverageB
    - kWheelSync * (angleB - angleA);
```

ここで、

```cpp
kWheelSync = 1.20;
```

程度。

つまり本来の意図としては、

\[
e_A =
u_{\rm ref} - \omega_A
+ k_{\rm sync}(\phi_B-\phi_A)
\]

\[
e_B =
u_{\rm ref} - \omega_B
- k_{\rm sync}(\phi_B-\phi_A)
\]

と考えられる。

その後、

```cpp
eAI = eA * 0.01 + 0.99 * eAI;
eBI = eB * 0.01 + 0.99 * eBI;

uA = kpA * eA + kiA * eAI;
uB = kpB * eB + kiB * eBI;
```

でPWMへ変換している。

代表値：

```cpp
kpA = 0.025;
kpB = 0.030;

kiA = 0.040;
kiB = 0.050;
```

---

# 6. `angleSpeedAverageA/B` の実装上の問題

車輪速度そのものは、エンコーダから一応計算している。

```cpp
angleSpeedA =
    alpha * angleSpeedA
    + (1.0 - alpha)
      * (angleA - anglePrevA) / dt;

angleSpeedB =
    alpha * angleSpeedB
    + (1.0 - alpha)
      * (angleB - anglePrevB) / dt;
```

ここまでは問題ない。

しかし、その後の平均値計算が、

```cpp
angleSpeedAverageA +=
    angleSpeedAverageA
    / (controlTimeDeltaSec / angleSensorTimeDeltaSec);

angleSpeedAverageB +=
    angleSpeedAverageB
    / (controlTimeDeltaSec / angleSensorTimeDeltaSec);
```

となっている。

本来おそらく意図されていたのは、

```cpp
angleSpeedAverageA +=
    angleSpeedA
    / (controlTimeDeltaSec / angleSensorTimeDeltaSec);

angleSpeedAverageB +=
    angleSpeedB
    / (controlTimeDeltaSec / angleSensorTimeDeltaSec);
```

である。

つまり、**第一項が `angleSpeedAverageA` ではなく `angleSpeedA` であるべきだった可能性が高い。**

---

# 7. なぜ現在のコードでは平均速度が0になるのか

現在のコードでは、各制御周期の最後に

```cpp
angleSpeedAverageA = 0;
angleSpeedAverageB = 0;
```

へリセットしている。

その状態で、

```cpp
angleSpeedAverageA +=
    angleSpeedAverageA / 100;
```

を実行しても、

\[
0 + \frac{0}{100} = 0
\]

であり、何度繰り返しても0のまま。

したがって、

\[
\boxed{
angleSpeedAverageA \approx 0
}
\]

\[
\boxed{
angleSpeedAverageB \approx 0
}
\]

となっている。

---

# 8. 実際の下位制御はどうなっているか

本来は、

\[
e_A =
u_{\rm LQR} - \omega_A
+ k_{\rm sync}(\phi_B-\phi_A)
\]

であるべきところが、

\[
\omega_A \approx 0
\]

として扱われているため、実際には

\[
\boxed{
e_A \approx
u_{\rm LQR}
+ k_{\rm sync}(\phi_B-\phi_A)
}
\]

\[
\boxed{
e_B \approx
u_{\rm LQR}
- k_{\rm sync}(\phi_B-\phi_A)
}
\]

となっている。

つまり、**下位速度PI制御は実質的に成立していない。**

---

# 9. 実際の制御構造

コード上は、

```text
LQR
↓
目標車輪速度
↓
速度PI
↓
PWM
```

に見える。

しかし実態は、

```text
姿勢・ジャイロ・エンコーダ
        ↓
     オブザーバ
        ↓
     4状態LQR
        ↓
 u_WheelAngleSpeed
        ↓
左右車輪差補正
(kWheelSync)
        ↓
小さいP + リークI
        ↓
      PWM
        ↓
     モータ
```

である。

つまり、

\[
\boxed{
\text{LQR出力} \rightarrow \text{ほぼ直接PWM}
}
\]

に近い。

---

# 10. なぜこれで追従性が良いのか

これは非常に重要。

もし本当に速度フィードバックを有効にすると、

```text
LQR
↓
目標車輪速度
↓
速度誤差
↓
PI
↓
PWM
```

という追加の閉ループが入る。

このとき、

- エンコーダ速度推定遅れ
- フィルタ遅れ
- PI応答
- PWM→モータ応答
- 0.10〜0.11 s のデッドタイム
- 0.07 s のモータ一次遅れ

が重なる。

倒立振子は非常に速く倒れるため、この追加遅れが致命的になりやすい。

実機では、**速度ループをほぼバイパスしてLQR出力を直接PWMへ反映している方が、姿勢に対して速く反応できた**と考えられる。

---

# 11. 実機の不安定性が非常に速い可能性

当初は振り子周期を

```text
T_pendulum = 0.75 s
```

としていた。

しかし後の動画計測では、

```text
T_pendulum ≈ 0.45 s
```

程度の可能性が出てきた。

倒立振子の不安定極の代表値は、

\[
\lambda \approx \frac{2\pi}{T}
\]

なので、

### T = 0.75 s

\[
\lambda \approx 8.38\ {\rm s^{-1}}
\]

### T = 0.45 s

\[
\lambda \approx 13.96\ {\rm s^{-1}}
\]

したがって不安定化の代表時間は、

\[
\tau_{\rm unstable} \approx \frac{1}{\lambda}
\]

より、

```text
T=0.75 s → 約0.119 s
T=0.45 s → 約0.072 s
```

となる。

実機の入力デッドタイムが

```text
0.10〜0.11 s
```

程度なので、

\[
\boxed{
\text{入力が効き始める前に、車体がかなり倒れる}
}
\]

非常に厳しい系である。

この条件では、下位速度ループによる追加遅れは特に不利。

---

# 12. 理論モデルとの食い違い

Pythonの設計モデルでは、アクチュエータを概念的に、

\[
u
\rightarrow
\text{6段デッドタイム近似}
\rightarrow
\frac{1}{0.07s+1}
\rightarrow
\omega
\]

としている。

つまり、

```text
u = 車輪速度指令
```

に近い意味として扱っている。

一方、実機では、

\[
u_{\rm LQR}
\rightarrow
小さいP+I
\rightarrow
PWM
\rightarrow
モータ
\]

に近い。

したがって、

\[
\boxed{
\text{設計モデルの入力 }u
\text{ と実機の }u_{\rm WheelAngleSpeed}
\text{ の意味が一致していない}
}
\]

可能性が高い。

---

# 13. `z1..z6` 直接フィードバックが悪かった理由

理論上の10状態LQRでは、

\[
z_1,\ldots,z_6
\]

を直接フィードバックする。

しかし実機ではこれを有効にすると、追従性が大きく悪化した。

理由として考えられるのは以下。

## 13.1 z状態は実測ではない

`z1..z6` はセンサから直接観測しているわけではなく、

- デッドタイム
- モータ時定数
- 入力履歴
- モデル

から推定している仮想状態。

したがってモデル誤差に敏感。

---

## 13.2 実デッドタイムとモデルが一致していない

モデル：

```text
約0.10 s
```

実測：

```text
約0.11 s
```

制御周期が10 msなので、これは約1サンプル分の差。

倒立系ではこの1サンプルが大きい。

---

## 13.3 zフィードバックは未来に効く入力を先回りして抑える

例えば正方向入力を過去に出していると、

\[
z_i > 0
\]

となる。

LQRで \(K_{z_i}>0\) なら、

\[
-K_{z_i}z_i < 0
\]

なので、現在の正方向入力を弱める。

理想モデルなら、

> 「過去に出した入力がもうすぐ効くから、今の入力を少し抑えよう」

という合理的な動作。

しかし実機モデルがずれていると、

> 「まだモータが反応していないのに、制御器だけがもう効くと思って入力を弱める」

ことになる。

これが、

```text
姿勢にモータ動作が追従しない
```

という症状につながった可能性が高い。

---

# 14. Δu-LQR が悪かった理由

入力差分

\[
\Delta u_k = u_k-u_{k-1}
\]

にペナルティを持たせた拡大LQRも試した。

狙いは、

```text
+u → -u → +u → -u
```

という細かいギコギコを抑えることだった。

しかし実機では、

```text
DU制御は姿勢にほとんど追従しなかった
```

という結果。

原因候補：

- 4状態制御から11状態制御へ一気に変更した
- `z1..z6` を直接使った
- 前回入力を状態化した
- Δu制約が応答を鈍らせた
- 0.10〜0.11 s のデッドタイムが大きいため、入力を滑らかにしすぎると姿勢追従が間に合わない

---

# 15. 平衡点近傍だけゲインを弱める方法も悪化した

別案として、

```cpp
u = raw_u * controlScale;
```

として、直立近傍では

```text
controlScale ≈ 0.75
```

までゲインを落とす方法も試した。

しかし、

```text
ギコギコが悪化
```

した。

理由として、

- 平衡点近傍でこそ線形LQRが最も有効
- そこでゲインを落とすと安定余裕が減る
- ゲインスケジュールが非線形切替として働く
- 0.11 s のデッドタイムに対して反応が鈍くなる

などが考えられる。

---

# 16. Rを増やす方法は良かった

一方、LQRの入力重み

\[
R
\]

を増やす方法は実機で有効だった。

実験結果：

```text
R=1.0
→ かなり機敏
→ ギコギコ大きめ

R=1.2
→ ギコギコ低減
→ 安定性良好

R=1.4
→ さらに良好
→ 現在のベスト
```

この方法は、実行時に単純にゲインをスケーリングするのではなく、

\[
J=
\sum
\left(
x^TQx + u^TRu
\right)
\]

を再設計するため、閉ループ全体として整合している。

---

# 17. 現時点のベスト構造

現在、最も良かった構成は以下。

## 上位

```text
4状態LQR
R = 1.4
```

制御則：

\[
u =
-K_\theta\hat{\theta}
-K_{\dot{\theta}}\hat{\dot{\theta}}
-K_\phi\hat{\phi}
-K_\omega\hat{\omega}
\]

代表ゲイン：

```cpp
K_theta       = -359.683463;
K_theta_dot   =  -42.940531;
K_wheel_angle =   -0.765166;
K_wheel_speed =   -5.268757;
```

## 遅延状態

```text
z1..z6
```

は、

- モデル
- オブザーバ

の内部では使う。

ただし、

\[
\boxed{
制御入力には直接入れない
}
\]

## 下位

実質：

\[
e_A \approx u_{\rm LQR} + k_{\rm sync}(\phi_B-\phi_A)
\]

\[
e_B \approx u_{\rm LQR} - k_{\rm sync}(\phi_B-\phi_A)
\]

その後、

```text
小さいP + リークI → PWM
```

---

# 18. `angleSpeedAverage` を直せば良いのか？

コード上の意味としては、

```cpp
angleSpeedAverageA +=
    angleSpeedA / N;
```

に直すのが正しい。

ただし、これを単純に直すと、

\[
e_A =
u_{\rm LQR}
-\omega_A
+k_{\rm sync}\Delta\phi
\]

になり、**本当に速度閉ループが有効になる。**

これはシステム構造の大変更。

したがって、

\[
\boxed{
「バグだから直す」＝「性能が良くなる」
とは限らない
}
\]

実際、過去にちゃんと速度追従させた構成では追従性が悪化した。

---

# 19. もし車輪速度フィードバックを再導入するなら

単純に、

```text
LQR → 速度PI → PWM
```

へ変更するのではなく、

\[
\boxed{
PWM =
K_{\rm ff}u_{\rm LQR}
+
K_p(u_{\rm LQR}-\omega)
+
K_i\int(u_{\rm LQR}-\omega)dt
}
\]

のようにする案が有力。

つまり、

## フィードフォワード

\[
K_{\rm ff}u_{\rm LQR}
\]

で現在の機敏な直接駆動を残す。

## 速度PI

\[
K_p e_\omega + K_i\int e_\omega dt
\]

は弱い補正として使う。

これなら、

- 現在の速い姿勢追従
- 実車輪速度補正
- モータ左右差補正

を両立できる可能性がある。

---

# 20. 重要な設計上の解釈

現在の実機結果を見る限り、

\[
\boxed{
「完全な理論モデルに忠実な制御」
より
「実測姿勢に直接強く反応する制御」
の方が良い
}
\]

可能性が高い。

特にこのロボットは、

- デッドタイムが大きい
- 倒立不安定性が速い
- アクチュエータモデル誤差が大きい
- エンコーダ速度推定に遅れがある
- PWM・摩擦・モータ特性が非線形
- z状態がモデル依存

という条件がある。

そのため、4状態だけの実測寄りフィードバックが結果的に**ロバスト制御的**に働いている可能性がある。

---

# 21. 今後の推奨方針

現時点では、以下を優先する。

1. **R=1.4 の4状態LQRを基準として固定**
2. `z1..z6` は直接フィードバックしない
3. Δu制御は一旦使わない
4. 平衡点近傍だけゲインを弱める方式は使わない
5. `angleSpeedAverage` は、評価目的でのみ修正版を試す
6. 速度FBを使う場合はフィードフォワード主体にする
7. `u_WheelAngleSpeed → PWM → 実車輪速度` の実伝達特性を改めて実測する
8. 実測した伝達関数を次のモデルへ使う
9. 振り子周期は 0.45 s 前後の可能性があり、以前の 0.75 s から再同定する
10. 実測デッドタイム 0.11 s を次回設計で再評価する

---

# 22. 他AIへ伝えるべき最重要ポイント

最重要事項だけ短くまとめると：

> この倒立振子ロボでは、コード上は「LQR出力→車輪速度PI→PWM」の二重ループに見えるが、`angleSpeedAverageA/B` の実装ミスにより実際の車輪速度フィードバックはほぼ0になっている。そのため実態は「4状態LQR→左右同期補正→小さいP+リークI→PWM」という直接駆動に近い。
>
> この“速度FBがほぼ無い”状態の方が、実機では最も姿勢追従性が良かった。
>
> `z1..z6` の遅延状態を直接フィードバックした完全10状態LQRや、Δuペナルティ付きLQRでは、モータが姿勢に追従しなくなった。
>
> 一方、4状態構造を維持したままLQRの入力重みRを1.0→1.2→1.4と上げると、ギコギコが低減し、R=1.4が現在のベストとなった。
>
> したがって今後は、完全な理論モデルへ無理に戻すより、4状態LQR＋直接PWM寄りの高速経路を維持し、必要なら弱い速度PIを補助的に追加する方向が有力。

---

# 23. 参考：現在の実質的な制御ブロック

```text
 BNO055 theta, theta_dot
         │
 Encoder phi
         │
         ▼
     Observer
         │
         ▼
  4-state LQR
  R = 1.4
         │
         ▼
 u_WheelAngleSpeed
         │
         ├───────────────┐
         │               │
         ▼               ▼
 + kSync(phiB-phiA)   - kSync(phiB-phiA)
         │               │
         ▼               ▼
 small P + leaky I   small P + leaky I
         │               │
         ▼               ▼
       PWM A           PWM B
         │               │
         ▼               ▼
      Motor A          Motor B
```

実車輪速度 `omegaA/B` は現状この下位ループへ有効に入っていない。

---

# 24. 参考：本来意図されていた可能性が高い構造

```text
LQR
 ↓
wheel speed reference
 ↓
measured wheel speed
 ↓
speed error
 ↓
PI
 ↓
PWM
```

しかし、この構造へ戻すと追加遅れが生じるため、倒立系では追従性が悪化する可能性が高い。

---

# 25. 結論

現状の実験結果は、

\[
\boxed{
\text{この実機では「理論的に完全な制御」より「遅れの少ない直接的な制御」が重要}
}
\]

ことを示している。

特に、

\[
\boxed{
4状態LQR + R=1.4 + z直接FBなし
}
\]

が現在の最良構成。

`angleSpeedAverageA/B` のバグは確かに存在するが、それを直すことは単なるバグ修正ではなく、制御構造そのものを

```text
LQR → ほぼ直接PWM
```

から

```text
LQR → 速度閉ループ → PWM
```

へ変更することを意味する。

そのため、修正する場合は必ず別系統として比較評価する必要がある。
